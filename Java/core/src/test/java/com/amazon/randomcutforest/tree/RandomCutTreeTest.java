/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazon.randomcutforest.tree;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.containsInAnyOrder;
import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.nullValue;
import static org.hamcrest.Matchers.sameInstance;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.IMultiVisitorFactory;
import com.amazon.randomcutforest.IVisitorFactory;
import com.amazon.randomcutforest.anomalydetection.AnomalyScoreVisitor;
import com.amazon.randomcutforest.config.Config;
import com.amazon.randomcutforest.imputation.ImputeVisitor;
import com.amazon.randomcutforest.sampler.Weighted;

public class RandomCutTreeTest {

    private static final double EPSILON = 1e-8;

    private Random rng;
    private RandomCutTree tree;

    @BeforeEach
    public void setUp() {
        rng = mock(Random.class);
        tree = RandomCutTree.builder().random(rng).centerOfMassEnabled(true).storeSequenceIndexesEnabled(true).build();

        // Create the following tree structure (in the second diagram., backticks denote
        // cuts)
        // The leaf point 0,1 has mass 2, all other nodes have mass 1.
        //
        // /\
        // / \
        // -1,-1 / \
        // / \
        // /\ 1,1
        // / \
        // -1,0 0,1
        //
        //
        // 0,1 1,1
        // ----------*---------*
        // | ` | ` |
        // | ` | ` |
        // | ` | ` |
        // -1,0 *-------------------|
        // | |
        // |```````````````````|
        // | |
        // -1,-1 *--------------------
        //
        // We choose the insertion order and random draws carefully so that each split
        // divides its parent in half.
        // The random values are used to set the cut dimensions and values.

        assertEquals(tree.getConfig(Config.BOUNDING_BOX_CACHE_FRACTION), 1.0);
        tree.setBoundingBoxCacheFraction(0.4);
        IVisitorFactory<Double> visitorFactory = (tree, x) -> new AnomalyScoreVisitor(tree.projectToTree(x),
                tree.getMass());
        assertThrows(IllegalStateException.class, () -> tree.traverse(new double[] { 0, 1 }, visitorFactory));
        int[] missingIndices = new int[] { 1 };
        IMultiVisitorFactory<double[]> multiVisitorFactory = (tree, y) -> new ImputeVisitor(y, tree.projectToTree(y),
                missingIndices, tree.projectMissingIndices(missingIndices), 1.0);
        assertThrows(IllegalStateException.class,
                () -> tree.traverseMulti(new double[] { 1, Double.NaN }, multiVisitorFactory));

        tree.addPoint(new double[] { -1, -1 }, 1);

        when(rng.nextDouble()).thenReturn(0.625);
        tree.addPoint(new double[] { 1, 1 }, 2);

        when(rng.nextDouble()).thenReturn(0.5);
        tree.addPoint(new double[] { -1, 0 }, 3);

        when(rng.nextDouble()).thenReturn(0.25);
        tree.addPoint(new double[] { 0, 1 }, 4);

        // add mass to 0,1
        tree.addPoint(new double[] { 0, 1 }, 5);

        tree.traverseMulti(new double[] { 1, Double.NaN }, multiVisitorFactory);
        tree.traverseMulti(new double[] { 0, Double.NaN }, multiVisitorFactory);
    }

    @Test
    public void testBuilderWithCustomerArguments() {
        RandomCutTree customTree = RandomCutTree.builder().random(rng).centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true).build();
        assertTrue(customTree.centerOfMassEnabled());
        assertTrue(customTree.storeSequenceIndexesEnabled());
    }

    @Test
    public void testDefaultTree() {
        RandomCutTree defaultTree = RandomCutTree.defaultTree();
        assertFalse(defaultTree.centerOfMassEnabled());
        assertFalse(defaultTree.storeSequenceIndexesEnabled());
    }

    /**
     * Verify that the tree has the form described in the setUp method.
     */
    @Test
    public void testInitialTreeState() {
        Node node = tree.getRoot();
        BoundingBox expectedBox = new BoundingBox(new double[] { -1, -1 }).getMergedBox(new double[] { 1, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertEquals(node.getCut().toString(), "Cut(1, -0.500000)");
        assertTrue(tree.leftOf(new double[] { -100, -1 }, node));
        assertThat(node.getCut().getDimension(), is(1));
        assertThat(node.getCut().getValue(), closeTo(-0.5, EPSILON));
        assertThat(node.getMass(), is(5));
        assertArrayEquals(new double[] { -0.2, 0.4 }, node.getCenterOfMass(), EPSILON);
        assertThat(node.getLeftChild().isLeaf(), is(true));
        assertThat(node.getLeftChild().getLeafPoint(), is(new double[] { -1, -1 }));
        assertThat(node.getLeftChild().getMass(), is(1));
        assertThat(node.getLeftChild().getSequenceIndexes(), contains(1L));

        node = node.getRightChild();
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 1, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(0));
        assertThat(node.getCut().getValue(), closeTo(0.5, EPSILON));
        assertThat(node.getMass(), is(4));
        assertArrayEquals(new double[] { 0.0, 0.75 }, node.getCenterOfMass(), EPSILON);
        assertThat(node.getRightChild().isLeaf(), is(true));
        assertThat(node.getRightChild().getLeafPoint(), is(new double[] { 1, 1 }));
        assertThat(tree.getLeafBoxFromLeafNode(node.getRightChild()),
                is(new BoundingBox(tree.getPoint(node.getRightChild()), new double[] { 1, 1 })));
        assertThat(node.getRightChild().getMass(), is(1));
        assertThat(node.getRightChild().getSequenceIndexes(), contains(2L));

        node = node.getLeftChild();
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 0, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(0));
        assertThat(node.getCut().getValue(), closeTo(-0.5, EPSILON));
        assertThat(node.getMass(), is(3));
        assertArrayEquals(new double[] { -1.0 / 3, 2.0 / 3 }, node.getCenterOfMass(), EPSILON);
        assertThat(node.getLeftChild().isLeaf(), is(true));
        assertThat(node.getLeftChild().getLeafPoint(), is(new double[] { -1, 0 }));
        assertThat(node.getLeftChild().getMass(), is(1));
        assertThat(node.getLeftChild().getSequenceIndexes(), contains(3L));
        assertThat(node.getRightChild().isLeaf(), is(true));
        assertThat(node.getRightChild().getLeafPoint(), is(new double[] { 0, 1 }));
        assertThat(node.getRightChild().getMass(), is(2));
        assertThat(node.getRightChild().getSequenceIndexes(), containsInAnyOrder(4L, 5L));

        tree.deletePoint(new double[] { 0, 1 }, 5);
        tree.addPoint(new double[] { 0, 1 }, 5);
        tree.addPoint(new double[] { 0, 1 }, 5);
        tree.deletePoint(new double[] { 0, 1 }, 5);
        assertThat(node.getRightChild().getSequenceIndexes(), containsInAnyOrder(4L, 5L));

        IVisitorFactory<Double> visitorFactory = (tree, x) -> new AnomalyScoreVisitor(tree.projectToTree(x),
                tree.getMass());
        assertEquals(tree.traverse(new double[] { 0, 1 }, visitorFactory), 0.451, 0.001);
        int[] missingIndices = new int[] { 1 };
        IMultiVisitorFactory<double[]> multiVisitorFactory = (tree, y) -> new ImputeVisitor(y, tree.projectToTree(y),
                missingIndices, tree.projectMissingIndices(missingIndices), 1.0);
        assertArrayEquals(tree.traverseMulti(new double[] { 0, Double.NaN }, multiVisitorFactory),
                new double[] { 0, 1 }, EPSILON);

        tree.addPoint(new double[] { 0, 0.75 }, 6);

        assertArrayEquals(tree.traverseMulti(new double[] { 1, Double.NaN }, multiVisitorFactory),
                new double[] { 1, 1 }, EPSILON);

    }

    @Test
    public void testDeletePointWithLeafSibling() {
        tree.deletePoint(new double[] { -1, 0 }, 3);

        // root node bounding box and cut remains unchanged, mass and centerOfMass are
        // updated

        Node node = tree.getRoot();
        BoundingBox expectedBox = new BoundingBox(new double[] { -1, -1 }).getMergedBox(new double[] { 1, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(1));
        assertThat(node.getCut().getValue(), closeTo(-0.5, EPSILON));
        assertThat(node.getMass(), is(4));
        assertArrayEquals(new double[] { 0.0, 0.5 }, node.getCenterOfMass());
        assertThat(node.getLeftChild().isLeaf(), is(true));
        assertThat(node.getLeftChild().getLeafPoint(), is(new double[] { -1, -1 }));
        assertThat(node.getLeftChild().getMass(), is(1));
        assertThat(node.getLeftChild().getSequenceIndexes(), contains(1L));

        // sibling node moves up and bounding box recomputed

        node = node.getRightChild();
        expectedBox = new BoundingBox(new double[] { 0, 1 }).getMergedBox(new double[] { 1, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(0));
        assertThat(node.getCut().getValue(), closeTo(0.5, EPSILON));
        assertEquals(node.getCut().getValue(), node.getCutValue());
        assertThat(node.getMass(), is(3));
        assertArrayEquals(new double[] { 1.0 / 3, 1.0 }, node.getCenterOfMass());
        assertThat(node.getLeftChild().isLeaf(), is(true));
        assertThat(node.getLeftChild().getLeafPoint(), is(new double[] { 0, 1 }));
        assertThat(node.getLeftChild().getMass(), is(2));
        assertThat(node.getLeftChild().getSequenceIndexes(), containsInAnyOrder(4L, 5L));
        assertThat(node.getRightChild().isLeaf(), is(true));
        assertThat(node.getRightChild().getLeafPoint(), is(new double[] { 1, 1 }));
        assertThat(node.getRightChild().getMass(), is(1));
        assertThat(node.getRightChild().getSequenceIndexes(), contains(2L));
    }

    @Test
    public void testDeletePointWithNonLeafSibling() {
        tree.deletePoint(new double[] { 1, 1 }, 2);

        // root node bounding box recomputed

        Node node = tree.getRoot();
        BoundingBox expectedBox = new BoundingBox(new double[] { -1, -1 }).getMergedBox(new double[] { 0, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(1));
        assertThat(node.getCut().getValue(), closeTo(-0.5, EPSILON));
        assertThat(node.getLeftChild().isLeaf(), is(true));
        assertThat(node.getLeftChild().getLeafPoint(), is(new double[] { -1, -1 }));
        assertThat(node.getLeftChild().getMass(), is(1));
        assertThat(node.getLeftChild().getSequenceIndexes(), contains(1L));

        // sibling node moves up and bounding box stays the same

        node = node.getRightChild();
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 0, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(0));
        assertThat(node.getCut().getValue(), closeTo(-0.5, EPSILON));
        assertThat(node.getLeftChild().isLeaf(), is(true));
        assertThat(node.getLeftChild().getLeafPoint(), is(new double[] { -1, 0 }));
        assertThat(node.getLeftChild().getMass(), is(1));
        assertThat(node.getLeftChild().getSequenceIndexes(), contains(3L));
        assertThat(node.getRightChild().isLeaf(), is(true));
        assertThat(node.getRightChild().getLeafPoint(), is(new double[] { 0, 1 }));
        assertThat(node.getRightChild().getMass(), is(2));
        assertThat(node.getRightChild().getSequenceIndexes(), containsInAnyOrder(4L, 5L));
    }

    @Test
    public void testDeletePointWithMassGreaterThan1() {
        tree.deletePoint(new double[] { 0, 1 }, 4);

        // same as initial state except mass at 0,1 is 1

        Node node = tree.getRoot();
        BoundingBox expectedBox = new BoundingBox(new double[] { -1, -1 }).getMergedBox(new double[] { 1, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(1));
        assertThat(node.getCut().getValue(), closeTo(-0.5, EPSILON));
        assertThat(node.getMass(), is(4));
        assertArrayEquals(new double[] { -0.25, 0.25 }, node.getCenterOfMass(), EPSILON);
        assertThat(node.getLeftChild().isLeaf(), is(true));
        assertThat(node.getLeftChild().getLeafPoint(), is(new double[] { -1, -1 }));
        assertThat(node.getLeftChild().getMass(), is(1));
        assertThat(node.getLeftChild().getSequenceIndexes(), contains(1L));

        node = node.getRightChild();
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 1, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(0));
        assertThat(node.getCut().getValue(), closeTo(0.5, EPSILON));
        assertThat(node.getMass(), is(3));
        assertArrayEquals(new double[] { 0.0, 2.0 / 3 }, node.getCenterOfMass(), EPSILON);
        assertThat(node.getRightChild().isLeaf(), is(true));
        tree.setLeafPointReference(node.getRightChild(), null); // should do nothing
        assertThat(node.getRightChild().getLeafPoint(), is(new double[] { 1, 1 }));
        assertThat(node.getRightChild().getMass(), is(1));
        assertThat(node.getRightChild().getSequenceIndexes(), contains(2L));

        node = node.getLeftChild();
        expectedBox = new BoundingBox(new double[] { -1, 0 }).getMergedBox(new double[] { 0, 1 });
        assertThat(node.getBoundingBox(), is(expectedBox));
        assertThat(node.getCut().getDimension(), is(0));
        assertThat(node.getCut().getValue(), closeTo(-0.5, EPSILON));
        assertThat(node.getMass(), is(2));
        assertArrayEquals(new double[] { -0.5, 0.5 }, node.getCenterOfMass());
        assertThat(node.getLeftChild().isLeaf(), is(true));
        assertThat(node.getLeftChild().getLeafPoint(), is(new double[] { -1, 0 }));
        assertThat(node.getLeftChild().getMass(), is(1));
        assertThat(node.getLeftChild().getSequenceIndexes(), contains(3L));
        assertThat(node.getRightChild().isLeaf(), is(true));
        assertThat(node.getRightChild().getLeafPoint(), is(new double[] { 0, 1 }));
        assertThat(node.getRightChild().getMass(), is(1));
        assertThat(node.getRightChild().getSequenceIndexes(), contains(5L));

        tree.addToBox(node, node.getRightChild().getLeafPoint());
        assertThat(node.getBoundingBox(), is(expectedBox));
    }

    @Test
    public void testDeleteRoot() {
        RandomCutTree tree = RandomCutTree.defaultTree();
        double[] point = new double[] { -0.1, 0.1 };
        tree.addPoint(point, 1);
        tree.deletePoint(point, 1);

        assertThat(tree.getRoot(), is(nullValue()));
    }

    @Test
    public void testDeleteChildOfRoot() {
        RandomCutTree tree = RandomCutTree.defaultTree();
        double[] point1 = new double[] { -0.1, 0.2 };
        double[] point2 = new double[] { -0.3, 0.4 };
        tree.addPoint(point1, 1);
        tree.addPoint(point2, 2);
        tree.deletePoint(point1, 1);

        Node root = tree.getRoot();
        assertThat(root.isLeaf(), is(true));
        assertThat(root.getLeafPoint(), is(point2));
    }

    @Test
    public void testDeletePointInvalid() {
        // specified sequence index does not exist
        assertThrows(IllegalStateException.class, () -> tree.deletePoint(new double[] { -1, 0 }, 99));

        // point does not exist in tree
        assertThrows(IllegalStateException.class, () -> tree.deletePoint(new double[] { -1.01, 0.01 }, 3));
    }

    @Test
    public void testAddPointToEmptyTree() {
        RandomCutTree tree = RandomCutTree.defaultTree();
        double[] point = new double[] { 111, -111 };
        tree.addPoint(point, 1);
        assertArrayEquals(point, tree.getRoot().getLeafPoint());
    }

    @Test
    public void testRandomCut() {
        // construct a box with side lengths 10, 0, and 30
        // random numbers generated between 0 and 0.25 should get mapped to dimension 0
        // random numbers generated between 0.25 and 1.0 should get mapped to dimension
        // 2

        BoundingBox box = new BoundingBox(new double[] { 0.0, 0.0, 0.0 })
                .getMergedBox(new double[] { 10.0, 0.0, 30.0 });

        when(rng.nextDouble()).thenReturn(0.0);
        Cut cut = tree.randomCut(rng, box);
        assertThat(cut.getDimension(), is(0));
        assertThat(cut.getValue(), is(0.0));

        when(rng.nextDouble()).thenReturn(0.1);
        cut = tree.randomCut(rng, box);
        assertThat(cut.getDimension(), is(0));
        assertThat(cut.getValue(), closeTo(10.0 * 0.1 / 0.25, EPSILON));

        when(rng.nextDouble()).thenReturn(0.25);
        cut = tree.randomCut(rng, box);
        assertThat(cut.getDimension(), is(0));
        assertThat(cut.getValue(), closeTo(10.0, EPSILON));

        when(rng.nextDouble()).thenReturn(0.4);
        cut = tree.randomCut(rng, box);
        assertThat(cut.getDimension(), is(2));
        assertThat(cut.getValue(), closeTo(30.0 * (0.4 - 0.25) / 0.75, EPSILON));

        when(rng.nextDouble()).thenReturn(0.99);
        cut = tree.randomCut(rng, box);
        assertThat(cut.getDimension(), is(2));
        assertThat(cut.getValue(), closeTo(30.0 * (0.99 - 0.25) / 0.75, EPSILON));

        when(rng.nextDouble()).thenReturn(1.0);
        cut = tree.randomCut(rng, box);
        assertThat(cut.getDimension(), is(2));
        assertThat(cut.getValue(), closeTo(30.0, EPSILON));
    }

    @Test
    public void testReplaceNode() {
        Node node1 = new Node(new double[] { 1.5, 2.7 });
        Node node2 = new Node(new double[] { 3, 1.1 });
        Node node3 = new Node(new double[] { 4, 1.1 });

        BoundingBox parentBox = new BoundingBox(new double[] { 1.5, 2.7 }).getMergedBox(new double[] { 3, 1.1 });
        Cut cut = new Cut(1, 2.0);
        Node parent = new Node(node1, node2, cut, parentBox);
        node1.setParent(parent);
        node2.setParent(parent);

        assertThat(node3.getParent(), is(nullValue()));

        // replace the left child
        tree.replaceChild(parent, node1, node3);
        assertThat(parent.getLeftChild(), sameInstance(node3));
        assertThat(parent.getRightChild(), sameInstance(node2));
        assertThat(node3.getParent(), sameInstance(parent));
        assertThat(node2.getParent(), sameInstance(parent));

        // reset
        parent = new Node(node1, node2, cut, parentBox);
        node1.setParent(parent);
        node2.setParent(parent);
        node3.setParent(null);

        // replace the right child
        tree.replaceChild(parent, node2, node3);
        assertThat(parent.getLeftChild(), sameInstance(node1));
        assertThat(parent.getRightChild(), sameInstance(node3));
        assertThat(node1.getParent(), sameInstance(parent));
        assertThat(node3.getParent(), sameInstance(parent));

        assertThrows(IllegalStateException.class, () -> tree.replaceChild(node2, node3, null));
    }

    @Test
    public void testGetSibling() {
        Node left = new Node(new double[] { 0.0, 0.0 });
        Node right = new Node(new double[] { 1.0, 1.0 });
        BoundingBox parentBox = new BoundingBox(left.getLeafPoint()).getMergedBox(right.getLeafPoint());
        Node parent = new Node(left, right, new Cut(0, 0.5), parentBox);

        left.setParent(parent);
        right.setParent(parent);

        assertThat(tree.getSibling(left), sameInstance(right));
        assertThat(tree.getSibling(right), sameInstance(left));
    }

    @Test
    public void testGetSiblingNullParent() {
        Node node = new Node(new double[] { 0.0, 0.0 });
        assertThrows(NullPointerException.class, () -> tree.getSibling(node));
    }

    @Test
    public void testGetSiblingMalformedTree() {
        Node left = new Node(new double[] { 0.0, 0.0 });
        Node right = new Node(new double[] { 1.0, 1.0 });
        BoundingBox parentBox = new BoundingBox(left.getLeafPoint()).getMergedBox(right.getLeafPoint());

        // the parent node does not link to `left`
        Node parent = new Node(null, right, new Cut(0, 0.5), parentBox);

        left.setParent(parent);
        right.setParent(parent);

        assertThrows(IllegalStateException.class, () -> tree.getSibling(left));
    }

    @Test
    public void testRandomSeedAndBoundingBoxes() {
        // two trees created with the same random number generator should exhibit
        // identical behavior
        long randomSeed = 1234567890L;
        RandomCutTree tree1 = RandomCutTree.defaultTree(randomSeed);
        RandomCutTree tree2 = RandomCutTree.defaultTree(randomSeed);
        RandomCutTree tree3 = RandomCutTree.defaultTree(randomSeed * 2);
        tree2.setBoundingBoxCacheFraction(0);

        double[] point1 = new double[] { 0.1, 108.4, -42.2 };
        double[] point2 = new double[] { -0.1, 90.6, -30.7 };

        tree1.addPoint(point1, 1L);
        tree1.addPoint(point2, 2L);

        tree2.addPoint(point1, 1L);
        tree2.addPoint(point2, 2L);

        tree3.addPoint(point1, 1L);
        tree3.addPoint(point2, 2L);

        Cut cut1 = tree1.getRoot().getCut();
        Cut cut2 = tree2.getRoot().getCut();
        Cut cut3 = tree3.getRoot().getCut();

        assertEquals(cut1.getDimension(), cut2.getDimension());
        assertEquals(cut1.getValue(), cut2.getValue());

        assertEquals(tree1.getRoot().getBoundingBox(), tree2.getRoot().getBoundingBox());

        assertNotEquals(cut1.getDimension(), cut3.getDimension());
        assertNotEquals(cut1.getDimension(), cut3.getDimension());

        // should succeed since sequence numbers are stored
        // should not delete the point
        tree3.deleteSequenceIndex(tree3.getRoot().getLeftChild(), 0L);
        assertThrows(IllegalArgumentException.class, () -> tree3
                .constructBoxInPlace(tree3.getRoot().getRightChild().getBoundingBox(), tree3.getRoot().getLeftChild()));
        assertEquals(tree3.constructBoxInPlace(tree3.getMutableLeafBoxFromLeafNode(tree3.getRoot().getRightChild()),
                tree3.getRoot().getLeftChild()), tree1.getBoundingBox(tree1.root));
    }

    @Test
    public void testUpdatesOnSmallBoundingBox() {
        // verifies on small bounding boxes random cuts and tree updates are functional
        RandomCutTree tree = RandomCutTree.defaultTree();

        List<Weighted<double[]>> points = new ArrayList<>();
        points.add(new Weighted<>(new double[] { 48.08 }, 0, 1L));
        points.add(new Weighted<>(new double[] { 48.08000000000001 }, 0, 2L));

        tree.addPoint(points.get(0).getValue(), points.get(0).getSequenceIndex());
        tree.addPoint(points.get(1).getValue(), points.get(1).getSequenceIndex());

        for (int i = 0; i < 10000; i++) {
            Weighted<double[]> point = points.get(i % points.size());
            tree.deletePoint(point.getValue(), point.getSequenceIndex());
            tree.addPoint(point.getValue(), point.getSequenceIndex());
        }
    }
}
