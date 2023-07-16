- when the matrix is fits the memory / thread count
    - k = target_element_count / thread count
    - each thread perform k of calculation required for a target element
- when the matrix is too large
    - load parts of tA and tB into GPU, calc and store temporarily
    - finish calculating all ele in tA and tB, get the one target element
    - repeat above to finish the total calculation

- Other optimization
    - tranpose matB, so that both matrices can operate on row-basis, which is more r/w friendly