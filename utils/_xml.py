

def extract_data(xml_content, tag):
    try:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start_index = xml_content.find(start_tag) + len(start_tag)
        end_index = xml_content.find(end_tag)
        data = xml_content[start_index:end_index]
    except Exception as e:
        data = None
        print(f"Error extracting XLSX file: {e}")
    return data
