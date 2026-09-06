import csv


file_path = "DATASETS/comtrade_furniture.csv"


with open(
    file_path,
    "r",
    encoding="cp1252",
    newline=""
) as file:

    reader = csv.reader(file)

    header = next(reader)
    first_row = next(reader)


print("\nHEADER COLUMN COUNT:")
print(len(header))

print("\nFIRST DATA ROW COLUMN COUNT:")
print(len(first_row))


print("\nFIRST 15 HEADER VALUES:")
for i, value in enumerate(header[:15]):
    print(i, value)


print("\nFIRST 15 DATA VALUES:")
for i, value in enumerate(first_row[:15]):
    print(i, value)


print("\nLAST 10 DATA VALUES:")
for i, value in enumerate(first_row[-10:]):
    print(i, value)