# MIMIC3 sql agent

# MIMIC-III SQL Agent with LangGraph

This is just my Toy project implements an interactive SQL agent that allows natural language querying over the MIMIC-III clinical database, built using the LangChain and LangGraph frameworks.
   
 Few things that i added: 
>- Integration with the MIMIC-III dataset (OMOP-CDM schema)  
>- Prompt specific for using MIMIC-III  
>- Interactive interface for iterative exploration and play

# Output example:

example #1<br>
Welcome to the SQL Assistant! <br>
Type 'exit' to quit.<br>
What would you like to know? how many patients are diagnosed with atrial fibrillation?<br>
================================== Ai Message ==================================

SELECT COUNT(DISTINCT person_id) <br>
FROM condition_occurrence <br>
JOIN concept ON condition_occurrence.condition_concept_id = concept.concept_id <br>
WHERE concept.concept_name = '%atrial fibrillation%';<br>
================================== Ai Message ==================================<br>

Answer: There are 10,276 patients diagnosed with atrial fibrillation.

===============================================<br>

example #2<br>
Welcome to the SQL Assistant! <br>
Type 'exit' to quit.<br>
What would you like to know? how many patients have diagnosed with hypertension?<br>

================================== Ai Message ==================================<br>

SELECT COUNT(DISTINCT person_id) 
FROM condition_occurrence 
WHERE condition_concept_id IN (
    SELECT concept_id 
    FROM concept 
    WHERE concept_name LIKE '%hypertension%';
================================== Ai Message ==================================<br>

Answer: There are 20,230 patients diagnosed with hypertension.

# Plans for further improvement
>- Add a query parsing node to interpret user input and decompose it into step-by-step executable tasks.
>- Extend the agent to support analytical workflows by implementing downstream code for research-driven data analysis.

# References
 >- [LangChain SQL Tutorial](https://python.langchain.com/docs/tutorials/sql_qa/)  
> - [Building a SQL Agent with LangGraph (Medium)](https://medium.com/@hayagriva99999/building-a-powerful-sql-agent-with-langgraph-a-step-by-step-guide-part-2-24e818d47672)
