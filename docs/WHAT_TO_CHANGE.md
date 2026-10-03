# What Needs to Change (Simple Version)

## The 3 big problems

1. **Nobody can use the model on a real Bitcoin transaction.**
   It needs 165 secret, pre-computed numbers that only exist in the Elliptic dataset.

2. **The model fails on the newest data.**
   On steps 43–49 it catches only 5 out of 169 bad transactions. That's barely better than guessing.

3. **The "reliable / degraded" label in the API is fake confidence.**
   It just looks at the step number the user sends. Send "35" and you get "reliable", whatever the transaction is.

## What to fix (in order)

1. **Fix the numbers that contradict each other.**
   The monitoring report says one thing in the text and another in its table. Pick one meaning and label it ("average per step" vs "all rows together").

2. **Finish the retraining experiment (notebook 09).**
   Show whether retraining on newer labels fixes the collapse, and how much the label delay hurts.

3. **Be honest about monitoring.**
   Most of your drift alarms go off even when the model is working fine. Say which alarms actually helped and which didn't.

4. **Rewrite the README.**
   - New story: "why fraud models break when the world changes".
   - Credit the original 2019 Elliptic paper, which already found the step-43 drop.
   - Add a "Limitations" section.

5. **Add error bars.**
   Some time steps have only 2–5 fraud cases, so those numbers are too small to trust alone.

6. **Make the API honest and safer.**
   - Rename the "confidence" thing so it says what it really is.
   - Show *why* each transaction was flagged (top 5 reasons).
   - Pin the xgboost version.
   - Make the Docker image smaller.
   - Fix CORS.

7. **(Big bonus) Score real transactions.**
   Build features you can compute yourself from public blockchain data, so the demo takes a real transaction ID.

8. **Clean up the docs.**
   19 files is too many. Cut them down to a README plus about 3 docs, and remove hype words like "catastrophic" and "proving".

## One-line summary

Your rigor is good. The story oversells it. Make it honest, finish the retraining experiment, and make it work on real data.

*Full step-by-step version for AI agents: `docs/IMPROVEMENT_PLAN.md`*
