# Add INSERT or DELETE or DROP keywords inspect by lizhe
QUERY_CHECKER = """
{query}
Double check the {dialect} query above for common mistakes, including:
- Using INSERT or DROP or DELETE which will edit origin database
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins


If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Only output ONE final SQL dialect in ONE line and no more explanation.

SQL Query: """
