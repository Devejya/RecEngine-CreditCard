def read_large_CSV(filepath, chunk):
    '''
    Returns the dataframe to be read using chunk size reading.
    Input: FilePath (Str), ChunkSize (Int)
    Output: DataFrame
    '''
    import pandas as pd
    df_chunk = pd.read_csv(filepath, chunksize=chunk)

    chunk_list = []  # append each chunk df here 

    # Each chunk is in df format
    for chunk in df_chunk:  
        # Once the data filtering is done, append the chunk to list
        chunk_list.append(chunk)
        
    # concat the list into dataframe 
    df = pd.concat(chunk_list)
    return df


def write_hdf(df, filepath, dfname):
    '''
    Write DataFrame as .h5 file to given FilePath with Filename
    Input: dataframe, filepath (Str), dfname (Str)
    OutPut: None
    Effects: Write to HDF 
    '''
    import pandas as pd 
    df.to_hdf(filepath+dfname+'.h5', 'df', format='t', complevel=5, complib='bzip2')
    return None


def from_hive(appName, dbname, table):
    '''
    Returns Pandas DataFrame read from a Hive Table
    Input: Str, Str, Str
    '''
    from pyspark.context import SparkContext
    from pyspark.sql import HiveContext, SparkSession
    import pandas as pd

    sc = SparkContext(appName=appName)
    # Optional creation of a HiveContext
    sql_context = HiveContext(sc)
    # Optional creation of a SparkSession
    spark = SparkSession(sc)
    spark = (SparkSession.builder.enableHiveSupport().getOrCreate())  

    spark_df = spark.read.table(dbname+"."+table)
    # Enable Arrow-based columnar data transfers
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    # Convert the Spark DataFrame to a pandas DataFrame using Arrow
    df = spark_df.select("*").toPandas()
    print(df.head())
    return df


def to_hive(df, appname, dbname, table, mode):
    '''
    Write Pandas dataframe to HIVE (Append or Overwrite)
    Input: Pd Dataframe, Str, Str, Str, Str
    Return: None
    Effect: Write data to HIVE 
    mode can be "append" or "overwrite"
    '''
    from pyspark.context import SparkContext
    from pyspark.sql import HiveContext, SparkSession
    import pandas as pd
    
    #Setting up spark sessions, etc. to be able to send data to HDFS (Hive)
    # Creating a SparkContext
    sc = SparkContext(appName=appname)
    # Optional creation of a HiveContext
    sql_context = HiveContext(sc)
    # Optional creation of a SparkSession
    spark = SparkSession(sc)
    spark = (SparkSession.builder.enableHiveSupport().getOrCreate())

    #Convert pandas dataframe to spark df & Save model results to HDFS (Hive)
    df_spark = spark.createDataFrame(df)
    df_spark.write.format('hive').mode(mode).saveAsTable(dbname+"."+table);

    return None


def read_hdf(filename):
    '''
    Read HDF5 (.h5) files into Pandas Dataframe
    '''
    import pandas as pd
    df = pd.read_hdf(filename, 'df')
    return df
