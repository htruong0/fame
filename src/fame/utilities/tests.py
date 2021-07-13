def check_relevant_permutations(permutations, num_jets):
    '''Check that the number of permutations and the actual permutations are correct.'''
    correct_length = {
        3: 2,
        4: 10,
        5: 27
    }
    correct_permutations = {
        3: [
            (1, 3, 2),
            (2, 3, 1)
        ],
        4: [
            (1, 3, 2),
            (1, 3, 4),
            (1, 4, 2),
            (1, 4, 3),
            (2, 3, 1),
            (2, 3, 4),
            (2, 4, 1),
            (2, 4, 3),
            (3, 4, 1),
            (3, 4, 2)
        ],
        5: [
            (1, 3, 2),
            (1, 3, 4),
            (1, 3, 5),
            (1, 4, 2),
            (1, 4, 3),
            (1, 4, 5),
            (1, 5, 2),
            (1, 5, 3),
            (1, 5, 4),
            (2, 3, 1),
            (2, 3, 4),
            (2, 3, 5),
            (2, 4, 1),
            (2, 4, 3),
            (2, 4, 5),
            (2, 5, 1),
            (2, 5, 3),
            (2, 5, 4),
            (3, 4, 1),
            (3, 4, 2),
            (3, 4, 5),
            (3, 5, 1),
            (3, 5, 2),
            (3, 5, 4),
            (4, 5, 1),
            (4, 5, 2),
            (4, 5, 3)
        ]
    }
    assert len(permutations) == correct_length[num_jets], "Number of permutations incorrect."
    assert all([perm1 == perm2 for perm1, perm2 in zip(permutations, correct_permutations[num_jets])]), "Not all permutations correct."

def check_phi_terms(phi_terms, num_jets):
    correct_shapes = {
        3: 0,
        4: (1, 1),
        5: (3, 3)
    }
    assert (phi_terms[0].shape[1], phi_terms[1].shape[1]) == correct_shapes[num_jets], "Number of phi terms incorrect."

def check_recoil_factors(recoil_factors, num_jets):
    correct_length = {
        3: 2,
        4: 10,
        5: 27
    }
    assert recoil_factors.shape[1] == correct_length[num_jets], "Number of recoil factors incorrect."

def check_all_dipoles(dipoles, num_jets):
    correct_length = {
        3: 2,
        4: 12,
        5: 33
    }
    assert dipoles.shape[1] == correct_length[num_jets], "Number of dipoles incorrect."
