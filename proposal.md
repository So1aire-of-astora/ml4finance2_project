# Proposal
## 1. Fake new detection
### Model Settings
    - Get feature vectors from body, headline, and/or joint (how?)
    - Concatnate
    - Feed into NN. Structure TBD

## 2. BTC & Stock price
### What we want to do
    - Super high frequency trading strategy?
    - Candidate features: RSI, KDJ, MACD, BOLL, since **all those indicators are bounded on [0, 1] and stationary**
    - Frequency may be set to per minute.
    - NN structure: probably lstm - baseline; attention based? s4 model?
    - Transaction costs

## 3. (Not considered) Sekiro game AI
This idea will not be taken into consideration since setting up the gaming environment is really challenging.
