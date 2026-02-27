import pandas as pd
import duckdb
import autogen
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "data.db"
TABLE_NAME = "dataset"

# ---------------- GLOBAL CHAT MEMORY ----------------
chat_history = []

# Safe DB Init
if os.path.exists(DB_PATH):
    try:
        duckdb.connect(DB_PATH).close()
    except:
        os.remove(DB_PATH)

db_conn = duckdb.connect(DB_PATH)

# ---------------- HELPER ----------------
def extract_text(response):
    try:
        text = response["content"]
        text = text.replace("```sql", "")
        text = text.replace("```python", "")
        text = text.replace("```", "")
        text = text.replace("sql", "")
        text = text.replace("python", "")
        return text.strip()
    except:
        return str(response)

# ---------------- LOAD DATA ----------------
def load_data(filepath):
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    db_conn.execute(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM df")
    return df

# ---------------- SCHEMA ----------------
def get_schema():
    schema = db_conn.execute(f"DESCRIBE {TABLE_NAME}").fetchdf()
    return schema.to_string()

# ---------------- SQL RUNNER ----------------
def run_sql(query):
    try:
        result = db_conn.execute(str(query)).fetchdf()
        return result.to_string()
    except Exception as e:
        return str(e)

# ---------------- EDA ----------------
def run_eda():
    try:
        df = db_conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 1000").fetchdf()

        return f"""
Rows: {len(df)}
Columns: {len(df.columns)}

Column Names:
{list(df.columns)}

Data Types:
{df.dtypes}

Sample Data:
{df.head().to_string()}
"""
    except Exception as e:
        return str(e)

# ---------------- QUERY TYPE ----------------
def detect_query_type(question):
    descriptive = [
        "describe","overview","summary",
        "tell me about","what is this dataset",
        "structure","columns","insight"
    ]
    for word in descriptive:
        if word in question.lower():
            return "EDA"
    return "SQL"

# ---------------- AUTOGEN CONFIG ----------------
api_key = os.getenv("GEMINI_API_KEY")

# Safety Check: Stop immediately if the .env file isn't loading the key
if not api_key:
    raise ValueError("🚨 GEMINI_API_KEY is missing! Python cannot find your .env file or the key is empty.")

config_list = [
    {
        "model": "gemini-2.5-flash", # <--- UPDATED THIS LINE
        "api_key": api_key,
        "api_type": "google"
    }
]

# ---------------- AGENTS ----------------
analyst = autogen.AssistantAgent(
    name="Analyst",
    llm_config={"config_list": config_list},
    system_message="""
Convert business question into analytical intent.

Identify:
Metric
Dimension
Calculation

Example:
Churn Rate → Percentage of Churn='Yes'

Output:
Metric:
Dimension:
Calculation:
"""
)

sql_agent = autogen.AssistantAgent(
    name="SQL_Generator",
    llm_config={"config_list": config_list},
    system_message="""
Generate SQL using intent and schema.

If calculation = percentage use:

COUNT(CASE WHEN condition THEN 1 END) * 100.0 / COUNT(*)

Table name = dataset

Return ONLY raw SQL.
Do NOT use ``` or markdown formatting.
"""
)

validator = autogen.AssistantAgent(
    name="Validator",
    llm_config={"config_list": config_list},
    system_message="Ensure SQL only uses valid columns. Respond VALID or INVALID"
)

explainer = autogen.AssistantAgent(
    name="Explainer",
    llm_config={"config_list": config_list},
    system_message="""
You must respond in exactly THREE sections:

1. Data-Backed Natural Language Answer
   - State the main finding clearly using the SQL result.

2. How the Answer Was Derived
   - Briefly explain how the metric was calculated 
     (e.g., count, percentage, grouping logic).

3. Business Insight
   - Provide short business interpretation.
   - If grouped results exist, compare groups.

Do NOT explain SQL errors.
Keep explanation concise and structured.
"""
)

eda_agent = autogen.AssistantAgent(
    name="EDA_Analyst",
    llm_config={"config_list": config_list},
    system_message="Explain dataset structure and insights"
)

viz_agent = autogen.AssistantAgent(
    name="Visualizer",
    llm_config={"config_list": config_list},
    system_message="""
You are a Data Science Expert. Your goal is to select the MOST appropriate visualization based on the user's intent and the data provided.

Selection Rules:
- Distribution/Comparison (e.g., Churn by Gender): Use a Bar Chart or Seaborn Countplot.
- Proportions (e.g., Market Share): Use a Pie Chart.
- Trends over time: Use a Line Chart.
- Relationships (e.g., Tenure vs Monthly Charges): Use a Scatter Plot.

Python Rules:
1. Use `matplotlib.pyplot as plt` and `seaborn as sns`.
2. Save strictly to 'chart.png' via `plt.savefig('chart.png', bbox_inches='tight')`.
3. ALWAYS include `plt.clf()` at the end.
4. Output ONLY valid Python code. No markdown.
"""
)

# ---------------- PIPELINE ----------------
def process_query(question):

    global chat_history

    query_type = detect_query_type(question)

    # Store user message
    chat_history.append({"role": "user", "content": question})

    if query_type == "EDA":
        eda = run_eda()

        eda_resp = eda_agent.generate_reply(
            messages=chat_history + [{"role": "user", "content": eda}]
        )

        explanation = extract_text(eda_resp)

        chat_history.append({"role": "assistant", "content": explanation})

        return {
            "intent":"EDA",
            "sql":"Not Required",
            "result":eda,
            "explanation":explanation,
            "has_viz": False
        }

    schema = get_schema()

    # Intent generation
    intent_resp = analyst.generate_reply(messages=chat_history)
    intent = extract_text(intent_resp)

    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:

        sql_resp = sql_agent.generate_reply(
            messages=chat_history + [{
                "role":"user",
                "content":f"""
Intent:
{intent}

Schema:
{schema}

If metric is churn rate:
Churn='Yes' means churned.
"""
            }]
        )

        sql_query = extract_text(sql_resp)
        result = run_sql(sql_query)

        # If success → exit loop
        if "Error" not in result and "Exception" not in result:
            break

        # If failed → feed error back
        chat_history.append({
            "role":"user",
            "content":f"SQL failed with error:\n{result}\nPlease fix the query."
        })

        retry_count += 1

    if retry_count == max_retries:
        return {
            "intent":intent,
            "sql":sql_query,
            "result":"SQL failed after retries",
            "explanation":result,
            "has_viz": False
        }

    explanation_resp = explainer.generate_reply(
        messages=chat_history + [{
            "role":"user",
            "content":f"SQL Result:\n{result}"
        }]
    )

    explanation = extract_text(explanation_resp)
    chat_history.append({"role":"assistant","content":explanation})

    # --- NEW VISUALIZATION LOGIC ---
    has_viz = False
    try:
        # Fetch the actual dataframe for visualization
        df = db_conn.execute(str(sql_query)).fetchdf()

        # only attempt a plot if the query returned anything sensible
        if not df.empty and (len(df) > 1 or len(df.columns) > 1):
            viz_prompt = (
                f"Intent: {intent}\n"
                f"Data snippet:\n{df.head(10).to_string()}\n"
                "Generate Python code to save this data as 'chart.png' with proper X/Y labels."
            )

            viz_resp = viz_agent.generate_reply(
                messages=[{"role": "user", "content": viz_prompt}]
            )
            viz_code = extract_text(viz_resp)

            # debug – print what the LLM gave us
            print("==== generated viz code ====")
            print(viz_code)
            print("==== end viz code ====")

            exec_env = {"df": df, "plt": plt, "sns": sns, "pd": pd}
            try:
                exec(viz_code, exec_env)
            except Exception as exec_err:
                print("visualization exec error:", exec_err)

            # if the agent forgot to save the figure, do it ourselves
            if not os.path.exists("chart.png"):
                try:
                    plt.savefig("chart.png", bbox_inches="tight")
                except Exception as save_err:
                    print("fallback save error:", save_err)
                finally:
                    plt.clf()

            if os.path.exists("chart.png"):
                has_viz = True

        # final fallback – a trivial bar chart of the first two columns
        if not has_viz and not df.empty:
            try:
                tmp = df.iloc[:, :2]
                if tmp.shape[1] >= 2:
                    tmp.plot(kind="bar", figsize=(6, 4))
                    plt.savefig("chart.png", bbox_inches="tight")
                    plt.clf()
                    has_viz = True
            except Exception as fallback_err:
                print("fallback plotting failed:", fallback_err)

    except Exception as e:
        print(f"Visualization generation skipped/failed: {e}")
        pass  # still return the text answer

    return {
        "intent": intent,
        "sql": sql_query,
        "result": result,
        "explanation": explanation,
        "has_viz": has_viz
    }

# ---------------- CHAINLIT WRAPPER ----------------
def create_agent(filepath):
    # Load the data and get the dataframe back
    df = load_data(filepath)
    
    # Generate a markdown preview of the first 5 rows
    # This allows the app to show the table immediately
    preview_markdown = df.head(5).to_markdown() 

    def agent_wrapper(query: str):
        response = process_query(query)
        
        # Wrap SQL in markdown code blocks for the "Copy" button
        formatted_sql = f"```sql\n{response['sql']}\n```"

        return {
            "output": f"### Answer\n{response['explanation']}\n\n### SQL Query\n{formatted_sql}\n\n### Raw Data Output\n```\n{response['result']}\n```",
            "has_viz": response["has_viz"]
        }

    # IMPORTANT: Return BOTH the wrapper and the preview string
    return agent_wrapper, preview_markdown