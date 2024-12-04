import ollama
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path

loaded_dfs = {}

def load_csv(file_path):
    try:
        data_path = Path('data') / file_path
        current_path = Path(file_path)
        
        if data_path.is_file():
            file_to_load = data_path
        elif current_path.is_file():
            file_to_load = current_path
        else:
            return f"Error: File not found in data/ or current directory: {file_path}"

        # Load the CSV
        df = pd.read_csv(file_to_load)
        name = file_to_load.stem  # Get filename without extension
        loaded_dfs[name] = df
        return f"Successfully loaded {name} dataset"

    except pd.errors.EmptyDataError:
        return "Error: The CSV file is empty"
    except pd.errors.ParserError:
        return "Error: Unable to parse CSV file - invalid format"
    except PermissionError:
        return "Error: Permission denied accessing the file"
    except Exception as e:
        return f"Error loading CSV: {str(e)}"

def scatter_plot(df, x_col, y_col):
    try:
        sns.scatterplot(data=df, x=x_col, y=y_col)
        plt.show()
    except Exception as e:
        return f"Error creating scatter plot: {e}"

def bar_plot(df, x_col, y_col):
    try:
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.show()
    except Exception as e:
        return f"Error creating bar plot: {e}"

def calculate_mean(df, col):
    try:
        return df[col].mean()
    except Exception as e:
        return f"Error calculating mean: {e}"

def calculate_sum(df, col):
    try:
        return df[col].sum()
    except Exception as e:
        return f"Error calculating sum: {e}"

custom_functions = {
    'load_csv': load_csv,
    'scatter_plot': scatter_plot,
    'bar_plot': bar_plot,
    'calculate_mean': calculate_mean,
    'calculate_sum': calculate_sum
}

available_functions = custom_functions

print('Available functions:', available_functions.keys())
print("\nExample commands you can try:")
print("1. 'Load the sales_data.csv file'")
print("2. 'Create a bar plot of product vs units from sales_data'")
print("3. 'Calculate the mean price from sales_data'")
print("4. 'Load the weather_data.csv file'")
print("5. 'Show a scatter plot of temperature vs humidity from weather_data'")

# Interactive chat loop
while True:
    try:
        user_input = input("\nEnter your request (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        response = ollama.chat(
            'llama3.2',
            messages=[{
                'role': 'user',
                'content': user_input,
            }],
            tools=[*available_functions.values()],
        )

        for tool in response.message.tool_calls or []:
            function_to_call = available_functions.get(tool.function.name)
            if function_to_call:
                print(f"Executing: {tool.function.name}")
                # Convert arguments to a dictionary if it's not already
                if isinstance(tool.function.arguments, dict):
                    args = tool.function.arguments
                else:
                    args = json.loads(tool.function.arguments)
                
                # Handle DataFrame references
                if 'df' in args and isinstance(args['df'], str):
                    df_name = args['df']
                    if df_name in loaded_dfs:
                        args['df'] = loaded_dfs[df_name]
                    else:
                        print(f"DataFrame {df_name} not found")
                        continue
                
                result = function_to_call(**args)
                print(f"Result: {result}")
            else:
                print('Function not found:', tool.function.name)

    except json.JSONDecodeError as e:
        print('Error parsing arguments:', e)
    except Exception as e:
        print('Error:', e)
        print('Response:', response.message)
