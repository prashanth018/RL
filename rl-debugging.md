Source: https://andyljones.com/posts/rl-debugging.html

Notes from RL debugging
- Use a seed
- Use huge batch size to cater to the stochasticity in the complete feedback loop
- Modularize the stack and swap out the components (like environments & algorithms) with simple and predictable ones to start with. Basically, reduce the stochasticity in the stack.
- Incrementally test your code by trying to cut the problem space into half.
- Do not debug the whole stack. You'll never be able to narrow down the problem. Testing and debugging components should take minutes.
- If you see an anomaly then chase it. Don't 'hope it goes away'. You might think the cool extra functionality you were planning to write today; adaptive reward scaling, a transformer; might just magically fix this anomalous behaviour. It won't. Give up the plans for the day and chase the anomaly instead.

