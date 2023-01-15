---
layout: default
title: Introduction
parent: Documentation
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---

# [](#header-1)Introduction to Symlearn

## [](#header-2)Main Concepts
Symbolic regression is a method that is concerned with identifying the mathematical form that generates output variables using a set of independent inputs through a particular system. The process starts by searching in the space of the different functional expression spaces and combining them to form a mathematical model. In contrast with other regression methods that have a predefined form that is used as an initial form to be optimized.

Symbolic regression methods use different mathematical expressions and combine them to form the structure of the solution. The used expressions include simple mathematical operators (addition, subtraction, multiplication, and division), trigonometric functions (sine, cosine, tangent, . . . , etc.), logical (and, or, not, nor, if-else, . . . , etc.), and others. Besides the functional expressions, it uses also different variables and constants as terminals. While terminals represent coefficients (in case the terminal is a constant) or one of the problem features (terminal is a variable), the functional expressions describe the relationship between the connected functional nodes and terminals. The combination of the functional and terminal sets creates the searching space which is defined by analysts as the following figure shows.

![](../../assets/images/functional_terminal.jpg)
*Functional and terminal sets (searching space set)*

---

## [](#header-2)Individual Representation

In ``symlearn``, each individual is a program that represents a solution for the problem. ``symlearn`` uses a hierarchical parse-tree structure that consists of connected functional expressions and terminals to represent the individuals as John Koza proposed in his first model the Genetic Programming (GP) as in following figure.

<img src="../../assets/images/parse_tree.jpg" width="250px"/>

*Individual representation in Symlearn*

Parse-tree structure is implemented easily as computer programs using the syntax of LISP which is also known as symbolic expression or S-expression. For instance, the computer program visualized in the previous figure is represented using S-expression as:

$$ + (*x3.14) (-7.6y) $$

which is equivalent to:

$$ (x \times 3.14) + (7.6 \times y) $$

---

## [](#header-2)Initialization

The optimization process starts by generating initial tree programs randomly using ``ramped half and half`` method which is a combination of two sub-methods; ``full`` and ``grow``. Using this forming method helps to
increase diversity and avoid similarity in the generated programs. In ``symlearn``, ``generate_population`` method initializes the first generation. 

---

## [](#header-2)Operations

### [](#header-3)Sharing

In ``sharing`` operation, a subtree instance is taken from the brighter tree and glued to an instance of the less bright tree. The subtree's instance is randomly selected, which increases the diversity in solution trees. The following figure shows an example of ``sharing`` operation.

<img src="../../assets/images/sharing.jpg" width="600px"/>

*Sharing operation*

### [](#header-3)Simplification

The ``simplification`` operation merges subtree nodes into one with an equivalent value, and it's applied only to branches that do not have any variable nodes. The following figure shows an example of the ``simplification`` operation where the expression 1 + 1 is replaced with only one node with a value of 2.

<img src="../../assets/images/simplification.jpg" width="600px"/>

*Simplification operation*

### [](#header-3)Substitution

Unlike the ``sharing`` operation which was described previously, the ``substitution`` operation happens in just one tree. In this operation, a random node is chosen and replaced with a suitable one To perform this without changing the structure of this node's subtree. Nodes are classified into different classes according to their arity: 0-arity nodes that represent terminal nodes, 1-arity nodes like trigonometric, logarithmic function nodes, 2-arity nodes like addition, subtraction, multiplication, and division expressions, or the power function nodes, and so on. In the ``substitution`` operation, the chosen node is replaced randomly with a new
node from the same class.

<img src="../../assets/images/substitution.jpg" width="600px"/>

*Substitution operation*
    