import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_community.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Load environment variables
load_dotenv()

# Database connection configuration
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"

connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def setup_database():
    """Set up sample database tables and data if they don't exist"""
    # Create a SQLDatabase instance without including it in the LangChain components yet
    db = SQLDatabase.from_uri(connection_string)
    return db


def basic_sql_chain(db: SQLDatabase):
    """Create a basic LangChain chain for SQL queries"""
    # Initialize the language model
    llm = OpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create a chain that converts natural language to SQL
    sql_chain = create_sql_query_chain(llm, db)
    
    # Set up a chain that executes the generated SQL and returns results
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    
    return sql_chain, db_chain


def advanced_sql_agent(db: SQLDatabase):
    """Create a more advanced LangChain SQL agent"""
    # Initialize the language model
    llm = OpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create toolkit and agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # Create the agent with the SQL toolkit
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return agent


def custom_query_chain(db: SQLDatabase):
    """Create a customized chain with specific prompts"""
    # Initialize the language model
    llm = OpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Get schema information once
    schema_info = db.get_table_info()
    
    # Custom prompt template for SQL generation
    sql_prompt = PromptTemplate.from_template(
        """You are a SQL expert. Given an input question, create a syntactically correct PostgreSQL query to run.
        Be careful to use only the column names that exist in the tables. Use appropriate joins as needed.
        
        Here is the database schema information:
        {schema}
        
        Human question: {question}
        
        SQL Query:"""
    )
    
    # Create a complete chain for SQL generation that includes the schema
    sql_generator_chain = (
        lambda x: sql_prompt.format(schema=schema_info, question=x)
    ) | llm | StrOutputParser()
    
    # Custom prompt for result explanation
    explanation_prompt = PromptTemplate.from_template(
        """You are a helpful data analyst. Based on the following SQL query and its results, provide a clear and concise
        explanation of the data. Highlight any important insights or patterns. If appropriate, suggest follow-up questions
        that might be valuable to explore.
        
        SQL Query: {query}
        
        Query Results: {results}
        
        Explanation:"""
    )
    
    # Function to execute SQL and format results
    def execute_query(sql: str) -> Dict[str, Any]:
        try:
            results = db.run(sql)
            return {"query": sql, "results": results}
        except Exception as e:
            return {"query": sql, "results": f"Error executing query: {str(e)}"}
    
    # Function to run the explanation chain
    def explain_results(query_results: Dict[str, Any]) -> str:
        explanation_chain = explanation_prompt | llm | StrOutputParser()
        return explanation_chain.invoke(query_results)
    
    return sql_generator_chain, execute_query, explain_results


def query_examples(db: SQLDatabase, db_chain, sql_chain, agent, custom_chains):
    """Run example queries using different LangChain components"""
    sql_generator_chain, execute_query, explain_results = custom_chains
    
    print("\n===== Example 1: Basic SQL Chain =====")
    question = "What is the average salary for employees in each department?"
    print(f"Question: {question}")
    
    # Generate SQL from natural language
    sql_query = sql_chain.invoke({"question": question})
    print(f"Generated SQL: {sql_query}")
    
    # Execute full chain
    result = db_chain.run(question)
    print(f"Result: {result}")
    
    print("\n===== Example 2: SQL Agent =====")
    question = "Which employees are working on the most projects and what are their roles?"
    print(f"Question: {question}")
    result = agent.run(question)
    print(f"Result: {result}")
    
    print("\n===== Example 3: Custom Chain with Explanation =====")
    question = "Find the departments with the highest total budget across all their projects"
    print(f"Question: {question}")
    
    # Generate SQL with the chain
    sql = sql_generator_chain.invoke(question)
    print(f"Generated SQL: {sql}")
    
    # Execute query
    query_results = execute_query(sql)
    print(f"Results: {query_results['results']}")
    
    # Generate explanation
    explanation = explain_results(query_results)
    print(f"Explanation: {explanation}")


def interactive_mode(agent):
    """Interactive mode for user to ask questions"""
    print("\n===== Interactive Mode =====")
    print("Ask questions about the employee database (type 'exit' to quit):")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() in ('exit', 'quit'):
            break
        
        try:
            result = agent.run(question)
            print("\nResult:")
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main function that runs all examples"""
    try:
        # Set up database
        print("Setting up database...")
        db = setup_database()
        
        print("\nInitializing LangChain components...")
        # Initialize different LangChain components
        sql_chain, db_chain = basic_sql_chain(db)
        agent = advanced_sql_agent(db)
        custom_chains = custom_query_chain(db)
        
        # Run example queries - passing db explicitly
        query_examples(db, db_chain, sql_chain, agent, custom_chains)
        
        # Enter interactive mode
        interactive_mode(agent)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()

# # Check if tables exist first
    # check_query = """
    # SELECT EXISTS (
    #     SELECT FROM information_schema.tables 
    #     WHERE table_name = 'employees'
    # );
    # """
    # tables_exist = db.run(check_query).strip() == 'True'
    
    # if not tables_exist:
    #     # Create sample tables and insert data
    #     create_tables_query = """
    #     -- Create departments table
    #     CREATE TABLE departments (
    #         department_id SERIAL PRIMARY KEY,
    #         department_name VARCHAR(100) NOT NULL,
    #         location VARCHAR(100)
    #     );
        
    #     -- Create employees table
    #     CREATE TABLE employees (
    #         employee_id SERIAL PRIMARY KEY,
    #         first_name VARCHAR(50) NOT NULL,
    #         last_name VARCHAR(50) NOT NULL,
    #         email VARCHAR(100) UNIQUE,
    #         hire_date DATE NOT NULL,
    #         salary NUMERIC(10, 2),
    #         department_id INTEGER REFERENCES departments(department_id),
    #         manager_id INTEGER REFERENCES employees(employee_id)
    #     );
        
    #     -- Create projects table
    #     CREATE TABLE projects (
    #         project_id SERIAL PRIMARY KEY,
    #         project_name VARCHAR(100) NOT NULL,
    #         start_date DATE,
    #         end_date DATE,
    #         budget NUMERIC(12, 2),
    #         department_id INTEGER REFERENCES departments(department_id)
    #     );
        
    #     -- Create employee_projects (many-to-many) table
    #     CREATE TABLE employee_projects (
    #         employee_id INTEGER REFERENCES employees(employee_id),
    #         project_id INTEGER REFERENCES projects(project_id),
    #         role VARCHAR(50),
    #         hours_allocated INTEGER,
    #         PRIMARY KEY (employee_id, project_id)
    #     );
        
    #     -- Insert sample departments
    #     INSERT INTO departments (department_name, location) VALUES
    #     ('Engineering', 'Building A'),
    #     ('Marketing', 'Building B'),
    #     ('Finance', 'Building C'),
    #     ('Human Resources', 'Building B'),
    #     ('Research', 'Building A');
        
    #     -- Insert sample employees
    #     INSERT INTO employees (first_name, last_name, email, hire_date, salary, department_id, manager_id) VALUES
    #     ('John', 'Smith', 'john.smith@company.com', '2015-06-15', 85000, 1, NULL),
    #     ('Sarah', 'Johnson', 'sarah.johnson@company.com', '2017-03-22', 92000, 1, 1),
    #     ('Michael', 'Williams', 'michael.williams@company.com', '2016-11-08', 78000, 2, NULL),
    #     ('Emily', 'Brown', 'emily.brown@company.com', '2018-09-30', 72000, 2, 3),
    #     ('David', 'Jones', 'david.jones@company.com', '2019-05-17', 67000, 3, NULL),
    #     ('Lisa', 'Garcia', 'lisa.garcia@company.com', '2020-02-12', 65000, 4, NULL),
    #     ('Robert', 'Miller', 'robert.miller@company.com', '2018-07-09', 71000, 5, NULL),
    #     ('Jennifer', 'Davis', 'jennifer.davis@company.com', '2021-01-20', 59000, 5, 7);
        
    #     -- Insert sample projects
    #     INSERT INTO projects (project_name, start_date, end_date, budget, department_id) VALUES
    #     ('Website Redesign', '2023-01-15', '2023-06-30', 120000, 1),
    #     ('Marketing Campaign Q2', '2023-04-01', '2023-06-30', 85000, 2),
    #     ('Financial Audit', '2023-03-10', '2023-05-15', 45000, 3),
    #     ('Employee Training Program', '2023-02-01', '2023-12-15', 75000, 4),
    #     ('Product Research', '2023-01-10', '2023-08-30', 250000, 5),
    #     ('Mobile App Development', '2023-05-01', '2023-11-30', 180000, 1);
        
    #     -- Assign employees to projects
    #     INSERT INTO employee_projects (employee_id, project_id, role, hours_allocated) VALUES
    #     (1, 1, 'Project Lead', 120),
    #     (2, 1, 'Developer', 160),
    #     (3, 2, 'Project Lead', 100),
    #     (4, 2, 'Marketing Specialist', 140),
    #     (5, 3, 'Financial Analyst', 160),
    #     (6, 4, 'Training Coordinator', 100),
    #     (7, 5, 'Research Lead', 140),
    #     (8, 5, 'Research Assistant', 160),
    #     (1, 6, 'Technical Advisor', 60),
    #     (2, 6, 'Lead Developer', 180);
    #     """
        
    #     db.run(create_tables_query)
    #     print("Sample database created successfully!")
    # else:
    #     print("Database tables already exist - skipping initialization.")