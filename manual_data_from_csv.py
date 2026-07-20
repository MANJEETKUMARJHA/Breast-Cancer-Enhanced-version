data = input("Enter your manual data:\n")

# Replace any number of spaces with a single comma
import re
result = re.sub(r"\s+", ",", data).strip(",")

print("coma added data:\n",result)