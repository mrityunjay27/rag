evaluation_data = [
    {
        "query": "What happens when an access token expires?",
        "expected_evidence": "Refresh tokens are used to obtain a new access token",
    },
    {
        "query": "How should refresh tokens be stored?",
        "expected_evidence": "Refresh tokens have a longer lifetime than access tokens and should be stored securely",
    },
    {
        "query": "What happens if the refresh token is invalid?",
        "expected_evidence": "If the refresh token is invalid or expired, the server rejects the request",
    },
]