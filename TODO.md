1. Jaynes-unaware programs with normal `rand` calls get intercepted, compiled to a version which works with the choice map interfaces - inference compilation automatic.

2. Automatic batching - majority/minority importance sampling + gradient corrections.

3. "Black box" logpdf interfaces - provide a macro which derives the correct Cassette overdub for any function call that the user wants.
