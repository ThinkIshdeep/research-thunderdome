# Demo walkthrough

The video is a 21-second captioned sequence of three screenshots captured from the V2 application. The QA host uses the same standalone view and engine. This is an edited walkthrough, not a continuous real-time recording.

## 1. Specify the experiment

Choose an installed sklearn benchmark, split seed and finite trial budget for three model families.

## 2. Reject a leakage fixture

The audit refuses preprocessing before splitting. The runner accepts only allowlisted configurations.

## 3. Run actual training trials

The leaderboard records validation scores and runtimes. The final test is evaluated after selecting the winner.
