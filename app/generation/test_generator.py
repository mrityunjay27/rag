from app.generation.generator import generate_answer


context = """
Refresh tokens are used to obtain a new access token
when the existing access token expires.

Refresh tokens have a longer lifetime than access tokens
and should be stored securely.
"""

query = "What are refresh tokens used for?"


answer = generate_answer(
    query=query,
    context=context,
)


print("Question:")
print(query)

print("\nAnswer:")
print(answer)