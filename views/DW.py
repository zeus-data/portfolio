import streamlit as st
import pandas as pd


st.title('Building an Azure Data Warehouse for Bike Share Data Analytics')

st.image('images/Chicago_Divvy_bikes.png')

st.write('''#### Project Overview
Divvy is a bike sharing program in Chicago, Illinois USA that allows riders
          to purchase a pass at a kiosk or use a mobile application to unlock
          a bike at stations around the city and use the bike for a specified amount of time.
          The bikes can be returned to the same station or to another station. The City of Chicago makes
          the anonymized bike trip data publicly available for projects like this where we can analyze the data.

 ''')
# st.write(' The Entity Relationship Diagram (ERD) looks like this:')
# st.image('images/dw/divvy-erd.png')

payment=pd.read_csv('dw_data/payments.csv', header=None )
rider=pd.read_csv('dw_data/riders.csv', header=None )
station=pd.read_csv('dw_data/stations.csv', header=None )
trip=pd.read_csv('dw_data/trips.csv', header=None )


st.write('''The available data consists of four tables, all of which are provided without headers:

**Payment**''')
st.dataframe(payment.head())
st.write('''        **Rider**''')
st.dataframe(rider.head())
st.write('''        **Station**''')
st.dataframe(station.head())
st.write('''        **Trip**''')
st.dataframe(trip.head())

st.write('---')





st.write('''The goal of this project is to develop a data warehouse solution using
          [Azure Synapse Analytics]("https://learn.microsoft.com/en-us/azure/synapse-analytics/get-started") and 
         [ELT]("https://www.ibm.com/topics/elt"). I will:
- Design a star schema based on the business outcomes listed below       
- **E**`xtract` the data on-prem PostgreSQL database
- **L**`oad` the data into Microsoft Synapse
- **T**`ransform` the data into the star schema''')

st.write('''Once the Data Warehouse solution is implemented, it will enable advanced analytics and data
          visualization, providing valuable insights to support business decisions.
          This solution will aggregate and organize the data to facilitate efficient reporting and deeper analysis.

The business outcomes I aim to achieve through this design are as follows:
1. Analyze how much time is spent per ride
- Based on date and time factors such as day of week and time of day
- Based on which station is the starting and / or ending station
- Based on age of the rider at time of the ride
- Based on whether the rider is a member or a casual rider
2. Analyze how much money is spent
- Per month, quarter, year
- Per member, based on the age of the rider at account start''')


st.write('''To set up the environment for this project, the **first step** is to create the data in PostgreSQL.
          This simulates the production environment where the data is utilized in an OLTP system.
          Therefore, I need to transfer these four CSV tables,
          currently stored on my local machine, into a PostgreSQL server and database running locally.
         The Python script below does that: ''')

code='''import psycopg

host = "localhost"
user = "postgres"
password = "*********"

# Create a new DB
sslmode = "disable"
dbname = "postgres"
conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
conn = psycopg.connect(conn_string)
conn.autocommit = True
print("Connection established")

cursor = conn.cursor()
cursor.execute('DROP DATABASE IF EXISTS udacityproject')
cursor.execute("CREATE DATABASE udacityproject")
# Clean up initial connection
conn.commit()
cursor.close()
conn.close()

# Reconnect to the new DB
dbname = "udacityproject"
conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
conn = psycopg.connect(conn_string)
conn.autocommit = True
print("Connection established")
cursor = conn.cursor()

# Helper functions
def drop_recreate(c, tablename, create):
    c.execute(f"DROP TABLE IF EXISTS {tablename};")
    c.execute(create)
    print(f"Finished creating table {tablename}")

def populate_table(c, filename, tablename):
    with open(filename, 'r') as f:
        try:
            c.copy(f"COPY {tablename} FROM STDIN WITH CSV HEADER DELIMITER AS ','", f)
            conn.commit()
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
            conn.rollback()
            c.close()
    print(f"Finished populating {tablename}")

# Create Rider table
table = "rider"
filename = './dw_data/riders.csv'
create = "CREATE TABLE rider (rider_id INTEGER PRIMARY KEY, first VARCHAR(50), last VARCHAR(50), address VARCHAR(100), birthday DATE, account_start_date DATE, account_end_date DATE, is_member BOOLEAN);"

drop_recreate(cursor, table, create)
populate_table(cursor, filename, table)

# Create Payment table
table = "payment"
filename = './dw_data/payments.csv'
create = "CREATE TABLE payment (payment_id INTEGER PRIMARY KEY, date DATE, amount MONEY, rider_id INTEGER);"

drop_recreate(cursor, table, create)
populate_table(cursor, filename, table)

# Create Station table
table = "station"
filename = './dw_data/stations.csv'
create = "CREATE TABLE station (station_id VARCHAR(50) PRIMARY KEY, name VARCHAR(75), latitude FLOAT, longitude FLOAT);"

drop_recreate(cursor, table, create)
populate_table(cursor, filename, table)

# Create Trip table
table = "trip"
filename = './dw_data/trips.csv'
create = "CREATE TABLE trip (trip_id VARCHAR(50) PRIMARY KEY, rideable_type VARCHAR(75), start_at TIMESTAMP, ended_at TIMESTAMP, start_station_id VARCHAR(50), end_station_id VARCHAR(50), rider_id INTEGER);"

drop_recreate(cursor, table, create)
populate_table(cursor, filename, table)

# Clean up
conn.commit()
cursor.close()
conn.close()

print("All done!")
'''
st.code(code)

st.write('''and a visual confirmation from the `command-line` that the database contains the 4 tables:''')

st.image("./images/dw/psql_creation.png")


st.write('''The **second step** is to design the star schema. After reviewing the business requirements
          and analyzing the four provided tables, I mapped the data and made the diagram below. 
         The data structure allows for efficient data reporting and analysis.

In the star schema, I used fact and dimension tables
          to organize the data. The fact table contains measurable, quantitative data, 
         typically representing transactions or events. The dimension tables provide context and descriptive information, 
         allowing the data in the fact table to be analyzed from different perspectives. ''')


st.image('./images/dw/star_schema.png')

st.write('''The **third step** is to transition to the cloud. It involves creating the necessary
          Microsoft Azure resources. Specifically, I need to set up an Azure Database for PostgreSQL and
          an Azure Synapse workspace.

An Azure Database for PostgreSQL is a fully managed relational database service that provides scalable 
         and secure hosting for PostgreSQL databases in the cloud. It handles database management tasks
          such as backups, patching, and scaling, allowing users to focus on application development.

An Azure Synapse workspace is an integrated analytics service that combines big data and data warehousing
          capabilities. It enables users to analyze large datasets, run queries, and manage data pipelines,
          providing powerful tools for business intelligence and data integration across different systems.''')

st.write(''' #### Azure Synapse''')
st.image('./images/dw/azure_synapse_1.png')
st.image('./images/dw/azure_synapse_2.png')
st.image('./images/dw/azure_synapse_3.png')

st.write(''' #### Azure PostgreSQL Database''')
st.image('./images/dw/azure_postgresql_SQL.png')


st.write('---')

st.write('''The **fourth step** is to extract the data from PostgreSQL. Within the Azure Synapse workspace, 
         I will utilize the ingest wizard to create a one-time pipeline that transfers the data from
          PostgreSQL into Azure Blob Storage.
          This process will convert all four tables into text files stored in Blob Storage, making them
          ready for loading into the data warehouse.

To accomplish this, I will first set up two linked services within the Synapse workspace.
          The first linked service will connect to my local PostgreSQL database, enabling
          data extraction. The second linked service will establish a connection to Azure Blob Storage,
          facilitating the transfer and storage of the extracted tables.''')

st.image('./images/dw/linked_service_1.png')
st.image('./images/dw/linked_service_2.png')
st.image('./images/dw/linked_service_3.png') 


st.write('''That's great.  After this step, there are now 4 text tables in the Azure blob storage.''')
st.image('./images/dw/linked_service_4.png')

st.write('''The **fifth step** is to create the staging tables within Azure Synapse.
          These tables are external tables, meaning they reference data stored externally,
          such as in Azure Blob Storage, rather than storing the data directly within the database. 
         These staging tables serve as the `Load` stage in the ELT process,
          where raw data is ingested and made available for further processing. At this stage, the focus
          is on ensuring that the data is accurately loaded from the source without applying any transformations.
          This approach allows for a clear separation between raw data ingestion and subsequent transformation
          logic. By leveraging external tables, the system can efficiently query and manage large datasets without
          duplicating data storage. Creating these staging tables is a critical step in preparing the data for
          transformation and integration into the final data warehouse schema.
          This ensures a structured and scalable workflow for handling data throughout the ELT pipeline.''')

st.image('./images/dw/load_process_1.png')

st.write('''and this is the SQL script for the 4 staging tables:''')

code='''IF NOT EXISTS (SELECT * FROM sys.external_file_formats WHERE name = 'SynapseDelimitedTextFormat') 
	CREATE EXTERNAL FILE FORMAT [SynapseDelimitedTextFormat] 
	WITH ( FORMAT_TYPE = DELIMITEDTEXT ,
	       FORMAT_OPTIONS (
			 FIELD_TERMINATOR = ',',
			 FIRST_ROW = 2,
			 USE_TYPE_DEFAULT = FALSE
			))
GO

IF NOT EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'project_full_project_full_dfs_core_windows_net') 
	CREATE EXTERNAL DATA SOURCE [project_full_project_full_dfs_core_windows_net] 
	WITH (
		LOCATION = 'abfss://project_full@project_full.dfs.core.windows.net' 
	)
GO

CREATE EXTERNAL TABLE dbo.staging_payment (
	[payment_id] bigint,
	[date] datetime2(0),
	[amount] float,
	[rider_id] bigint
	)
	WITH (
	LOCATION = 'publicpayment.txt',
	DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO


IF NOT EXISTS (SELECT * FROM sys.external_file_formats WHERE name = 'SynapseDelimitedTextFormat') 
	CREATE EXTERNAL FILE FORMAT [SynapseDelimitedTextFormat] 
	WITH ( FORMAT_TYPE = DELIMITEDTEXT ,
	       FORMAT_OPTIONS (
			 FIELD_TERMINATOR = ',',
			 FIRST_ROW = 2,
			 USE_TYPE_DEFAULT = FALSE
			))
GO

IF NOT EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'project_full_project_full_dfs_core_windows_net') 
	CREATE EXTERNAL DATA SOURCE [project_full_project_full_dfs_core_windows_net] 
	WITH (
		LOCATION = 'abfss://project_full@project_full.dfs.core.windows.net' 
	)
GO

CREATE EXTERNAL TABLE dbo.staging_rider (
	[rider_id] bigint,
	[first] nvarchar(4000),
	[last] nvarchar(4000),
	[address] nvarchar(4000),
	[birthday] datetime2(0),
	[account_start_date] datetime2(0),
	[account_end_date] datetime2(0),
	[is_member] bit
	)
	WITH (
	LOCATION = 'publicrider.txt',
	DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO


IF NOT EXISTS (SELECT * FROM sys.external_file_formats WHERE name = 'SynapseDelimitedTextFormat') 
	CREATE EXTERNAL FILE FORMAT [SynapseDelimitedTextFormat] 
	WITH ( FORMAT_TYPE = DELIMITEDTEXT ,
	       FORMAT_OPTIONS (
			 FIELD_TERMINATOR = ',',
			 FIRST_ROW = 2,
			 USE_TYPE_DEFAULT = FALSE
			))
GO

IF NOT EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'project_full_project_full_dfs_core_windows_net') 
	CREATE EXTERNAL DATA SOURCE [project_full_project_full_dfs_core_windows_net] 
	WITH (
		LOCATION = 'abfss://project_full@project_full.dfs.core.windows.net' 
	)
GO

CREATE EXTERNAL TABLE dbo.staging_station (
	[station_id] nvarchar(4000),
	[name] nvarchar(4000),
	[latitude] float,
	[longitude] float
	)
	WITH (
	LOCATION = 'publicstation.txt',
	DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO



IF NOT EXISTS (SELECT * FROM sys.external_file_formats WHERE name = 'SynapseDelimitedTextFormat') 
	CREATE EXTERNAL FILE FORMAT [SynapseDelimitedTextFormat] 
	WITH ( FORMAT_TYPE = DELIMITEDTEXT ,
	       FORMAT_OPTIONS (
			 FIELD_TERMINATOR = ',',
			 FIRST_ROW = 2,
			 USE_TYPE_DEFAULT = FALSE
			))
GO

IF NOT EXISTS (SELECT * FROM sys.external_data_sources WHERE name = 'project_full_project_full_dfs_core_windows_net') 
	CREATE EXTERNAL DATA SOURCE [project_full_project_full_dfs_core_windows_net] 
	WITH (
		LOCATION = 'abfss://project_full@project_full.dfs.core.windows.net' 
	)
GO

CREATE EXTERNAL TABLE dbo.staging_trip (
	[trip_id] nvarchar(4000),
	[rideable_type] nvarchar(4000),
	[start_at] datetime2(0),
	[ended_at] datetime2(0),
	[start_station_id] nvarchar(4000),
	[end_station_id] nvarchar(4000),
	[rider_id] bigint
	)
	WITH (
	LOCATION = 'publictrip.txt',
	DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
	FILE_FORMAT = [SynapseDelimitedTextFormat]
	)
GO


'''
st.code(code, language="SQL", line_numbers=True)

st.write('''The **sixth** and final step involves transforming the data into the star schema using
          the CREATE EXTERNAL TABLE AS SELECT (CETAS) statement. 
         This process takes the raw data stored in the staging tables and restructures it to match the
          final star schema design. CETAS is a powerful SQL feature that allows the creation 
         of new external tables based on the results of a query, combining data selection and
          table creation in a single operation. By applying this method, the data is efficiently
          transformed into the fact and dimension tables that make up the star schema. 
         These tables are optimized for analytical querying, enabling faster and more effective data analysis.
          The transformation ensures that the data is well-organized, with relationships between the fact and
          dimension tables clearly defined to support complex queries. Below are the SQL scripts I created
          to perform this transformation, mapping the data from the staging tables into the final star
          schema structure. This step is essential for
          preparing the data to meet business requirements and facilitate advanced
          analytics in the data warehouse.''')



st.write('''`dim_rider`''')
code='''IF OBJECT_ID ('dbo.dim_rider') IS NOT NULL
BEGIN
    DROP TABLE dbo.dim_rider;
END

CREATE EXTERNAL TABLE dim_rider
WITH   (
    LOCATION = 'dim_rider',
    DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
    FILE_FORMAT = [SynapseDelimitedTextFormat] 
)
AS

SELECT
 rider_id
,first AS first_name
,last AS last_name
,is_member
,birthday
,account_start_date
,account_end_date
FROM
dbo.staging_rider;
GO
'''
st.code(code, language="SQL", line_numbers=True)



st.write('''`dim_date`''')
code='''IF OBJECT_ID ('dbo.dim_date') IS NOT NULL
BEGIN
    DROP TABLE dbo.dim_date;
END

CREATE EXTERNAL TABLE dbo.dim_date
WITH   (
    LOCATION = 'dbo.dim_date',
    DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
    FILE_FORMAT = [SynapseDelimitedTextFormat] 
)
AS
 SELECT
 DISTINCT(CONVERT(NVARCHAR, date, 112)) as date_id
,date
,DATEPART(WEEKDAY, date) AS day_of_week
,DAY(date) AS day_of_month
,DATEPART(DAYOFYEAR, date) AS day_of_year
,MONTH(date) AS month
,DATEPART(QUARTER, date) AS quarter
,YEAR(date) AS year
FROM dbo.staging_payment;
GO
'''
st.code(code, language="SQL", line_numbers=True)


st.write('''`dim_station`''')
code='''IF OBJECT_ID ('dbo.dim_station') IS NOT NULL
BEGIN
    DROP TABLE dbo.dim_station;
END

CREATE EXTERNAL TABLE dbo.dim_station
WITH   (
    LOCATION = 'dbo.dim_station',
    DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
    FILE_FORMAT = [SynapseDelimitedTextFormat] 
)
AS
SELECT
  station_id
 ,name
 ,longitude
 ,latitude
FROM dbo.staging_station;
GO

'''
st.code(code, language="SQL", line_numbers=True)



st.write('''`fact_payment`''')
code='''IF OBJECT_ID ('dbo.fact_payment') IS NOT NULL
BEGIN
    DROP TABLE dbo.fact_payment;
END

CREATE EXTERNAL TABLE dbo.fact_payment
WITH   (
    LOCATION = 'dbo.fact_payment',
    DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
    FILE_FORMAT = [SynapseDelimitedTextFormat] 
)
AS

SELECT
payment_id
,amount
,DISTINCT(INT, CONVERT(NVARCHAR, date, 112)) as date_id
,rider_id
FROM dbo.staging_payment;
GO
'''
st.code(code, language="SQL", line_numbers=True)



st.write(''' and `fact_trip`''')
code='''IF OBJECT_ID ('dbo.fact_trip') IS NOT NULL
BEGIN
    DROP TABLE dbo.fact_trip;
END

CREATE EXTERNAL TABLE dbo.fact_trip
WITH   (
    LOCATION = 'dbo.fact_trip',
    DATA_SOURCE = [project_full_project_full_dfs_core_windows_net],
    FILE_FORMAT = [SynapseDelimitedTextFormat] 
)
AS

SELECT
    	 trip_id
	,rideable_type
	,DATEDIFF(MINUTE, t.start_at, t.ended_at) AS duration
	,start_station_id
	,end_station_id
	,r.rider_id
	,DATEDIFF(year, r.birthday, t.start_at) as rider_age
	,start_at
	,ended_at

FROM 
    dbo.staging_trip AS t
    JOIN
    dbo.staging_rider AS r
    ON
    t.rider_id = r.rider_id
    ;
GO
'''
st.code(code, language="SQL", line_numbers=True)


st.write('''In conclusion, after completing the transformation step using CREATE EXTERNAL TABLE AS SELECT (CETAS),
          the new system is fully equipped to provide valuable business insights by querying the newly created star
          schema tables using SQL. The process of designing and implementing the star schema, including organizing
          the data into fact and dimension tables, has optimized the structure for analytical queries and reporting.
          The final system, supported by the integration of Azure Synapse, Azure Blob Storage, and PostgreSQL,
          allows for efficient and scalable data handling. By leveraging SQL to query the star schema, users
          can now uncover actionable insights, such as ride durations segmented by time of day, starting and
          ending stations, rider demographics, and spending trends across different time periods. This enables
          the business to better understand customer behavior, improve operational decision-making, and identify
          growth opportunities. Overall, the project successfully transitions from raw data extraction to a robust
          data warehouse solution,
          ensuring that the data infrastructure supports comprehensive and meaningful analysis to meet business
          goals.''')