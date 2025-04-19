""" Generates list of intialized visualizers """
import pandas as pd
import seaborn as sns
from libs.dashboard import dashboard_app
import traceback

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

def dataframe_visualizer(code_imdict, dash_app):
    """Returns Dataframe visuualizers and list fo processed Daataframes """
    try:
        codedict = dict(code_imdict)
        list_dataframe_names = []

        for code in codedict.keys():
            try:
                loc = {}
                replace_str = ""
                command = "var = " + code.replace('"', "'") + replace_str
                print(f"Executing command: {command}")  # Debug print
                exec(command, globals(), loc)
                if "var" not in loc:
                    print("Error: 'var' not defined in local scope")
                    continue
                if not isinstance(loc["var"], pd.DataFrame):
                    print(f"Error: 'var' is not a DataFrame, got {type(loc['var'])}")
                    continue
                dash_app = dashboard_app(loc["var"], dash_app, {})
                list_dataframe_names.append("")
            except Exception as e:
                print(f"Error processing code: {code}")
                print(f"Error details: {str(e)}")
                print(traceback.format_exc())
                continue

        return list_dataframe_names, dash_app
    except Exception as e:
        print(f"Fatal error in dataframe_visualizer: {str(e)}")
        print(traceback.format_exc())
        return [], dash_app
