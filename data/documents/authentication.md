# Authentication

Our application uses JWT-based authentication.

## Access Tokens

When a user logs in successfully, the server generates an access token.
The client sends the access token with API requests using the Authorization header.

Access tokens are short-lived and should not be stored permanently.

## Refresh Tokens

Refresh tokens are used to obtain a new access token when the existing
access token expires.

Refresh tokens have a longer lifetime than access tokens and should be
stored securely.

## Token Refresh

The client sends the refresh token to the token refresh endpoint.
The server validates the refresh token and generates a new access token.

If the refresh token is invalid or expired, the server rejects the request.