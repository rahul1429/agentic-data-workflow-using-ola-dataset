import os
import pandas as  pd
import requests

class ETLTools:

    def __init__(self):
        pass

    def extract_data(self, url:str, output_folder:str, format:str) -> str:
        """
        Extracts data from the given (url) and saves it to the specified (output folder).

        Args:
            url (str): The URL to extract data from.
            output_folder (str): The folder where the extracted data will be saved.

        Returns:
            str: The path to the saved data file.
        """

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_folder = os.path.join(project_root, output_folder)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()  # Assuming the data is in JSON format
            filename = os.path.join(output_folder, f"extracted_data.{format}")
            os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

            df = pd.json_normalize(data['results'])  # Normalize the JSON data into a flat table
            if format == "json":
                df.to_json(filename, orient='records', lines=True)
            elif format == "csv":
                df.to_csv(filename, index=False)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}. Supported formats are 'json', 'csv', and 'parquet'.")

            return filename

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            return None



    def context_for_transform_load(self, file_path:str):
        """
        Transforms and loads data from the specified file path to the output folder.

        Args:
            file_path (str): The path to the input data file.
            output_folder (str): The folder where the transformed data will be saved.
            format (str): The format of the output data file.

        Returns:
            str: The top 3 rows of the transformed data as a string.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".json":
            df = pd.read_json(file_path, lines=True)
        elif file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Supported extensions are '.json', '.csv', and '.parquet'.")

        top_3_rows = str(df.head(3)) # string representation of the top 3 rows of the dataframe, since it will be used for context in the prompt for the LLM to generate the SQL query
        return top_3_rows


def execute_code(self, code:str):
    """
    Executes the provided code string in a local scope.

    Args:
        code (str): The code to be executed.
    
    Returns: str: The output of the executed code or an error message if execution fails.    
    """
    try:
        exec(code)
        return "Code executed successfully."
    except Exception as e:
        return f"Error executing code: {e}"


if __name__ == "__main__":
    obj = ETLTools()
    #print(obj.extract_data("https://pokeapi.co/api/v2/pokemon/", "data/extract", "json"))
    path = "/Users/rahulk/Downloads/Projects/OLA_Data_Agent/data/extract/extracted_data.json"
    print(obj.context_for_transform_load(path))