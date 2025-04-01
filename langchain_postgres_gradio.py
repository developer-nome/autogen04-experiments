import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit

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


def advanced_sql_agent(db: SQLDatabase):
    """Create a more advanced LangChain SQL agent"""
    # Initialize the language model
    llm = OpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create toolkit and agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return agent


def rag_query(user_input):
    db = setup_database()
    agent = advanced_sql_agent(db)
    result = agent.run(user_input)
    return result


def main():
    """Main function that runs all examples"""
    try:
        # # Initialize the Gradio interface
        # iface = gr.Interface(fn=rag_query,
        #                      inputs=[gr.Textbox(label="Question")],
        #                      outputs="text",
        #                      title="Company Projects Chat Bot",
        #                      description="Welcome",
        #                      flagging_mode="never")

        with gr.Blocks() as iface:
            with gr.Row():
                output_textbox = gr.Textbox(label="output")
            with gr.Row():
                input_question = gr.Textbox(label="Question")
            with gr.Row():
                btn_clear = gr.Button("Clear")
                btn_submit = gr.Button("Submit")
                btn_submit.variant = "primary"
                btn_submit.click(fn=rag_query, inputs=input_question, outputs=output_textbox)

        
        iface.launch()
        
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

            # CREATE TABLE IF NOT EXISTS public.employee_bonus (
            #     employee_id integer NOT NULL,
            #     bonus_amount numeric(10,2),
            #     payment_date date,
            #     CONSTRAINT employee_bonus_pkey PRIMARY KEY (employee_id),
            #     CONSTRAINT employee_bonus_employee_id_fkey FOREIGN KEY (employee_id)
            #         REFERENCES public.employees (employee_id) MATCH SIMPLE
            #         ON UPDATE NO ACTION
            #         ON DELETE NO ACTION
            # );
        
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

        # INSERT INTO employee_bonus (employee_id, bonus_amount, payment_date) VALUES
        # (1, 2300, '2024-01-15'),
        # (2, 1005, '2024-02-20'),
        # (3, 5000, '2024-03-25'),
        # (4, 4100, '2024-04-30'),
        # (5, 3300, '2024-05-10'),
        # (6, 2720, '2024-06-05'),
        # (7, 1200, '2024-07-01');
    #     """
        
    #     db.run(create_tables_query)
    #     print("Sample database created successfully!")
    # else:
    #     print("Database tables already exist - skipping initialization.")
