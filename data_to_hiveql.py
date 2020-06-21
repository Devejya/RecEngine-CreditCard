import pandas as pd 
import numpy as np 
import argparse
import os

###Function definitions
def setup_columns(df, table):
    '''
    returns a string with dataframe column names and respective HIVE dtypes AND string of tuple of all column names in dataframe
    '''

    list_cols = list(map(lambda x: x.lower(), list(df.columns)))
    list_col_types = []
    for col in list(df.columns):
        list_col_types.append(str(df[col].dtype))

    hiveql_col_types_dict = {
        'int64':'BIGINT',
        'int32':'INT',
        'int16':'INT',
        'int8':'INT',
        'uint8':'INT',
        'uint16':'INT',
        'uint32':'INT',
        'uint64':'INT',
        'object':'STRING',
        'float64': 'FLOAT',
        'float32': 'FLOAT',
        'datetime64':'TIMESTAMP',
        'bool': 'BOOLEAN'
    }

    list_hive_col_types = list(map(hiveql_col_types_dict.get, list_col_types))
    list_cols_with_types = [m+' '+n for m,n in zip(list_cols,list_hive_col_types)]
    final_str = ','.join(list_cols_with_types)

    #col_names = str(tuple(map(lambda l: '{0}.'.format(table)+l, list_cols))).replace("'","")
    col_names = str(tuple(list_cols)).replace("'","").lower()

    return final_str, col_names
    

def setup_values(df):
    '''
    returns a string of tuples with the values from the input dataframe
    '''
   #pandas null values are in (NaN) form, whereas, hiveQL requires NULL

    list_datatuples = list(map(str, list(df.itertuples(index=False, name=None))))
    str_data = ','.join(list_datatuples)

    return str_data.replace('nan','NULL')


def insert_values(str_data, table_name, db_name, col_names):
    '''
    returns the INSERT statement as STRING required with table and db name and the given data.
    '''

    return "INSERT INTO {0}.{1} {2} VALUES {3};".format(db_name,table_name, col_names, str_data)


def create_table_str(str_col, table_name, db_name, col_names):
    '''
    returns final hiveql command to drop, create new table and insert data as a string
    '''

    str_drop = 'drop table if exists {0}.{1};\n'.format(db_name, table_name)

    str_create = "CREATE TABLE IF NOT EXISTS {0}.{1} ({2});\n".format(db_name, table_name, str_col)

    final_str = str_drop+str_create

    return final_str


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Flags for which data to readin and put in HEAT')
    
    parser.add_argument('--file', dest = 'file',  action='store', type=str, required = True)

    parser.add_argument('--table', dest = 'table',  action='store', type=str, required = True)
    parser.add_argument('--db', dest = 'db',  action='store', type=str, required = True)

    parser.add_argument('--txt', dest = 'txt',  action='store', type=str, required = True)

    results = parser.parse_args()

    if not (os.path.isfile(results.file)):
        print ("Data file does NOT exist")
        sys.exit()

    #This can be changed to read_csv or read_excel or whatever you require
    if '.h5' in results.file:
        df = pd.read_hdf(results.file, 'df')

    #Do this for CSVs
    #Chunks done to deal with v.v.large dataframes
    if '.csv' in results.file:
        import time

        start = time.time()

        # read the large csv file with specified chunksize
        df_chunk = pd.read_csv(results.file, chunksize=100000)

        chunk_list = []  # append each chunk df here 

        # Each chunk is in df format
        for chunk in df_chunk:  
            # Once the data filtering is done, append the chunk to list
            chunk_list.append(chunk)
            
        # concat the list into dataframe 
        df = pd.concat(chunk_list)

        print("Reading in dataframe took %.2f seconds" % ((time.time() - start)))

    #Set up the required strings needed to write to the HEAT Hive table
    str_cols, cols = setup_columns(df, results.table)

    #Need to make insert statement in chunks due to large dataframe
    #chunk size needs to adjust for large and small dataframes
    #Total Number of data elements should be around ~1.8 mil
    columns = df.shape[1]
    n = 1400000//columns  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]
    #insert_values = []
    counter = 1

    print('Shape of DataFrame:', df.shape)

    for chunk in list_df:
        print ('Chunk Number: ', counter, '\n Out of Total Chunks: ', len(list_df))        
        #Set up the INSERT statement with all the values for the chunk
        insert_value = setup_values(chunk)
        insert_stuff = insert_values(insert_value, results.table, results.db, cols)
        #Write the INSERT statement to txt file 
        file = open(results.txt.strip('.txt')+str(counter)+'.txt',"w+")
        file.write(insert_stuff)
        file.close()
        counter+=1

    final_drop_create = create_table_str(str_cols, results.table, results.db, cols)

    #write HIVEQL command to txt file
    file = open(results.txt,"w")
    file.write(final_drop_create)
    file.close()
