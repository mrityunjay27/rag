from app.embeddings.embedder import generate_embedding


text = """
Refresh tokens are used to obtain a new access token
when the existing access token expires.
"""

embedding = generate_embedding(text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(embedding))
print("First 10 values:", embedding[:10])

""""
[-0.07267069816589355, -0.03717968985438347, -0.0022856665309518576, 0.032964784651994705, 0.02122998982667923, -0.024544328451156616, 0.035749711096286774, -0.03730691596865654, 0.09825892746448517, 0.005288269836455584]
"""