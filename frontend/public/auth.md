# auth.md

## Overview
Authentication and registration guide for agents using DivinityAI.

## Discovery
- OAuth Protected Resource Metadata: https://muslim.nevatal.id/.well-known/oauth-protected-resource
- OAuth 2.0 Authorization Server Metadata: https://muslim.nevatal.id/.well-known/oauth-authorization-server
- OpenID Connect Discovery: https://muslim.nevatal.id/.well-known/openid-configuration
- API Catalog: https://muslim.nevatal.id/.well-known/api-catalog
- Agent Card: https://muslim.nevatal.id/.well-known/agent-card.json

## Registration
Public content can be read without credentials. Agents that need write
access register at https://muslim.nevatal.id/api/agent/register and receive bearer credentials.
