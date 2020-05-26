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

package com.amazon.randomcutforest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.stream.Collector;

import com.amazon.randomcutforest.returntypes.ConvergingAccumulator;
import com.amazon.randomcutforest.tree.RandomCutTree;

public abstract class AbstractForestTraversalExecutor {

    protected final ArrayList<TreeUpdater> treeUpdaters;
    protected long totalUpdates;

    protected AbstractForestTraversalExecutor(ArrayList<TreeUpdater> treeUpdaters) {
        this.treeUpdaters = treeUpdaters;
        totalUpdates = 0;
    }

    public long getTotalUpdates() {
        return totalUpdates;
    }

    /**
     * Update the forest with the given point. The point is submitted to each
     * sampler in the forest. If the sampler accepts the point, the point is
     * submitted to the update method in the corresponding Random Cut Tree.
     *
     * @param point The point used to update the forest.
     */
    public void update(double[] point) {
        totalUpdates++;
        double[] pointCopy = Arrays.copyOf(point, point.length);
        update(pointCopy, totalUpdates);
    }

    /**
     * Internal update method which submits the given point and sequence index to
     * {@link TreeUpdater#update(double[], long)} for each TreeUpdater managed by
     * this executor.
     *
     * @param pointCopy     The point values that were are using the update the
     *                      forest. The name of this parameter is a reminder that we
     *                      should update the forest with a copy of the point that
     *                      was passed in, since the original point may be modified
     *                      later.
     * @param sequenceIndex The sequence index to assign to this point. This should
     *                      be a unique value for every point submitted.
     */
    protected abstract void update(double[] pointCopy, long sequenceIndex);

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A visitor is constructed for each tree using the visitor
     * factory, and then submitted to
     * {@link RandomCutTree#traverseTree(double[], Visitor)}. The results from all
     * the trees are combined using the accumulator and then transformed using the
     * finisher before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param accumulator    A function that combines the results from individual
     *                       trees into an aggregate result.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            BinaryOperator<R> accumulator, Function<R, S> finisher);

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A visitor is constructed for each tree using the visitor
     * factory, and then submitted to
     * {@link RandomCutTree#traverseTree(double[], Visitor)}. The results from
     * individual trees are collected using the {@link java.util.stream.Collector}
     * and returned. Trees are visited in parallel using
     * {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param collector      A collector used to aggregate individual tree results
     *                       into a final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            Collector<R, ?, S> collector);

    /**
     * Visit each of the trees in the forest sequentially and combine the individual
     * results into an aggregate result. A visitor is constructed for each tree
     * using the visitor factory, and then submitted to
     * {@link RandomCutTree#traverseTree(double[], Visitor)}. The results from all
     * the trees are combined using the {@link ConvergingAccumulator}, and the
     * method stops visiting trees after convergence is reached. The result is
     * transformed using the finisher before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param accumulator    An accumulator that combines the results from
     *                       individual trees into an aggregate result and checks to
     *                       see if the result can be returned without further
     *                       processing.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForest(double[] point, Function<RandomCutTree, Visitor<R>> visitorFactory,
            ConvergingAccumulator<R> accumulator, Function<R, S> finisher);

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A multi-visitor is constructed for each tree using the
     * visitor factory, and then submitted to
     * {@link RandomCutTree#traverseTreeMulti(double[], MultiVisitor)}. The results
     * from all the trees are combined using the accumulator and then transformed
     * using the finisher before being returned.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a multi-visitor.
     * @param accumulator    A function that combines the results from individual
     *                       trees into an aggregate result.
     * @param finisher       A function called on the aggregate result in order to
     *                       produce the final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForestMulti(double[] point,
            Function<RandomCutTree, MultiVisitor<R>> visitorFactory, BinaryOperator<R> accumulator,
            Function<R, S> finisher);

    /**
     * Visit each of the trees in the forest and combine the individual results into
     * an aggregate result. A multi-visitor is constructed for each tree using the
     * visitor factory, and then submitted to
     * {@link RandomCutTree#traverseTreeMulti(double[], MultiVisitor)}. The results
     * from individual trees are collected using the
     * {@link java.util.stream.Collector} and returned. Trees are visited in
     * parallel using {@link java.util.Collection#parallelStream()}.
     *
     * @param point          The point that defines the traversal path.
     * @param visitorFactory A factory method which is invoked for each tree to
     *                       construct a visitor.
     * @param collector      A collector used to aggregate individual tree results
     *                       into a final result.
     * @param <R>            The visitor result type. This is the type that will be
     *                       returned after traversing each individual tree.
     * @param <S>            The final type, after any final normalization at the
     *                       forest level.
     * @return The aggregated and finalized result after sending a visitor through
     *         each tree in the forest.
     */
    public abstract <R, S> S traverseForestMulti(double[] point,
            Function<RandomCutTree, MultiVisitor<R>> visitorFactory, Collector<R, ?, S> collector);
}
