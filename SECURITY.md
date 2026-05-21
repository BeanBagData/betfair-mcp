# Security notes

## Credential exposure in upstream git history

This repository (`BeanBagData/betfair-mcp` on GitHub) committed `credentials.json` to public history across multiple commits before reverting. The blobs are still recoverable from the public packfile. Anyone who pulled the public repo while those commits were live may have those values.

This local clone is not the upstream — we cannot rewrite the public history. The mitigations applied here are:

1. `credentials.json` is now in `.gitignore` and no longer read by the client code.
2. Credentials are loaded from environment variables (`BETFAIR_USERNAME`, `BETFAIR_PASSWORD`, `BETFAIR_APP_KEY`) read from a local `.env` file.
3. The user's working Betfair credentials are different from the values that were committed upstream, so no rotation is required for *this* user. If you are reading this and the committed values *are* your real credentials: rotate them at https://developer.betfair.com immediately.

## Credential setup

1. Copy `.env.example` to `.env`.
2. Paste your real Betfair username, password, and app key into `.env`.
3. Confirm `.env` is gitignored (`git status` should not show it).
4. Never commit `.env` or `credentials.json`.

## Paper mode

`PAPER_MODE=true` (default) prevents `place_lay_bet`, `place_back_bet`, and `cancel_order` from reaching Betfair. Flumine streaming is also blocked in paper mode because Flumine places live exchange orders through its own client. Iterate freely with `PAPER_MODE=true`; use polling mode for simulated execution. Only set `PAPER_MODE=false` when you intend to bet real money.
