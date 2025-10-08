import csv

def read_medical_data(csv_path):
    data = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # reads rows as dictionaries
        for row in reader:
            data.append({
                "title": row["title"],
                "source": row["source"],
                "url": row["url"],
                "content": row["content"]
            })
    return data

