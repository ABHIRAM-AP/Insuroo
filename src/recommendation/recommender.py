from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from data.models.user_profile import UserProfile, RecommendationResponse
import json

class PolicyRecommender:
    def __init__(self, rag_instance):
        """
        Initialize with an instance of InsuranceRAG to reuse the LLM and vectorstore.
        """
        self.rag = rag_instance
        self.llm = rag_instance.llm
        self.vectorstore = rag_instance.vectorstore
        
        # Specialized prompt for recommendation
        self.recommendation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Insurance Advisor for Indian Government Schemes.
Your goal is to analyze a user's profile and recommend the most suitable insurance policies from the provided context.

Policies available in context:
1. Ayushman Bharat (Health Insurance)
2. PMJJBY (Life Insurance)
3. PMFBY (Crop Insurance)

Rules:
- Be precise about eligibility (age, income, occupation).
- Provide a clear reasoning for each recommendation.
- List specific benefits relevant to the user.
- Output MUST be a valid JSON matching this exact structure:
{{
  "user_name": "Name of the user",
  "recommendations": [
    {{
      "policy_name": "Name of policy",
      "eligibility_status": "Highly Recommended / Eligible / Not Eligible",
      "reasoning": "Why this policy fits",
      "benefits": ["Benefit 1", "Benefit 2"]
    }}
  ],
  "summary": "Overall summary of advice"
}}

CONTEXT:
{context}"""),
            ("human", """User Profile:
Name: {name}
Age: {age}
Gender: {gender}
Occupation: {occupation}
Annual Income: {annual_income}
Farmer: {is_farmer}
Below Poverty Line: {is_below_poverty_line}
Pre-existing Conditions: {has_preexisting_conditions}
Location: {location}
Additional Info: {additional_info}

Please recommend the best policies for this user. Ensure the output is JUST the JSON and nothing else.""")
        ])

    def recommend(self, profile: UserProfile) -> dict:
        # 1. Create a search query based on profile
        search_query = f"Insurance scheme eligibility for {profile.age} year old {profile.occupation} earning {profile.annual_income}."
        if profile.is_farmer:
            search_query += " PMFBY Crop insurance."
        if profile.is_below_poverty_line:
            search_query += " Ayushman Bharat health insurance."
        if profile.additional_info:
            search_query += f" {profile.additional_info}"
        
        # 2. Retrieve relevant documents
        docs = self.vectorstore.similarity_search(search_query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # 3. Format prompt
        # We need to escape braces in JSON examples in the system prompt, or use F-string carefully.
        # Above I used double braces {{ }} which should work for .format()
        
        prompt_input = {
            "context": context,
            "name": profile.name,
            "age": profile.age,
            "gender": profile.gender,
            "occupation": profile.occupation,
            "annual_income": profile.annual_income,
            "is_farmer": profile.is_farmer,
            "is_below_poverty_line": profile.is_below_poverty_line,
            "has_preexisting_conditions": profile.has_preexisting_conditions,
            "location": profile.location,
            "additional_info": profile.additional_info or "None provided"
        }
        
        # 4. Get LLM response
        messages = self.recommendation_prompt.format_messages(**prompt_input)
        response = self.llm.invoke(messages)
        
        # 5. Parse JSON (handling potential formatting issues)
        content = response.content
        print(f"--- DEBUG: LLM RAW RESPONSE ---\n{content}\n--- END DEBUG ---")

        try:
            # Clean possible markdown formatting
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Find the first { and last } to be safe
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end+1]
            
            # Clean up trailing commas which often cause issues
            import re
            content = re.sub(r',\s*}', '}', content)
            content = re.sub(r',\s*]', ']', content)
                
            recommendation_data = json.loads(content)
            
            # Ensure mandatory fields exist (basic fallback)
            if "user_name" not in recommendation_data:
                recommendation_data["user_name"] = profile.name
            if "recommendations" not in recommendation_data:
                 recommendation_data["recommendations"] = []
            if "summary" not in recommendation_data:
                recommendation_data["summary"] = "No summary provided."
                
            return recommendation_data
        except Exception as e:
            print(f"Error parsing recommendation JSON: {e}")
            # Try a very simple fallback if parsing fails but name is known
            return {
                "user_name": profile.name,
                "recommendations": [],
                "summary": f"Could not parse recommendations JSON. Raw output: {content[:200]}..."
            }
