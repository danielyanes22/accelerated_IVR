import os
import sqlite3
import pandas as pd 

#initialise connection to database
db_path = 'data/liposome_IVR.db'

if os.path.isfile(db_path) is True:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        print(f"successfully connected to {db_path} and created cursor!")
    except sqlite3.Error as e:
        print(e)
else:
    print(f"{db_path} does not exist  - No DB created!")
    

#get list of formulation IDs in database
formulation_IDs = pd.read_sql("""SELECT 
                                    ID 
                                 FROM 
                                    formulation
                        """, conn)

formulation_ID_list = formulation_IDs['ID'].tolist()