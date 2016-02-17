import unittest

import quadedge as qe


class TestQuadedge(unittest.TestCase):
    def test_quadedge(self):
        q = qe.QuadEdge()
        e = q.base

        self.assertIs(e, q.edges[0])
        self.assertIsNot(e, e.rot)
        self.assertIs(e._rot, e.rot)
        self.assertIs(e, e.rot.rot.rot.rot)
        self.assertIs(e.rot.rot.rot, e.inv_rot)

        self.assertIsNot(e, e.sym)
        self.assertIs(e, e.sym.sym)

        self.assertIs(e.o_next, e)
        self.assertIs(e.o_prev, e)
        self.assertIs(e.d_next, e)
        self.assertIs(e.d_prev, e)

        self.assertIs(e.l_next, e.sym)
        self.assertIs(e.l_prev, e.sym)
        self.assertIs(e.r_next, e.sym)
        self.assertIs(e.r_prev, e.sym)

    def test_splice(self):
        q0 = qe.QuadEdge()
        q1 = qe.QuadEdge()

        e0 = q0.base
        e1 = q1.base

        self.assertIsNot(e0, e1)
        self.assertIs(e0.o_next, e0)

        qe.splice(e0, e1)

        self.assertIs(e0.o_next, e1)
        self.assertIs(e1.o_next, e0)

        self.assertIs(e0.l_next, e0.sym)
        self.assertIs(e0.l_prev, e1.sym)



if __name__ == '__main__':
    unittest.main()
