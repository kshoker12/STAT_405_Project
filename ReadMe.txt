
Joseph will be doing the data processing now.


Findings:

The subject metadata is structured as a forest. That is the subject ids are a a collection of trees whose root nodes are Science with id 1642 and Math with id 3. They both have parent id NULL.

Problem when you look at a question it has these subject ids that are also a tree not just the leaf nodes. We can isloate them though, and then do our inference as planned.
- That seems fine though as all of the 21 questions (out of 28 561). Thus a very small number of questions. These 21 questions are actually forests (Ie they have the math subject keys and the science subject ids which form a tree).

So now how are we going to use this for a bayesian model.?