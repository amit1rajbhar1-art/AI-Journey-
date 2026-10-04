from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
The terrace of my house is my favourite place. 
In the evening, the sky turns orange and pink.I gaze at the clouds and watch the sun set. 
I can see birds flying home and children playing in the street below. 
A cool breeze blows and I feel calm. When I am worried, I go up there and just breathe. 
It is my quiet corner in a busy city
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=10,
)  

chunks = splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")