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

package com.amazon.randomcutforest.imputation;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;
import java.util.Random;

import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.anomalydetection.AnomalyScoreVisitor;
import com.amazon.randomcutforest.tree.INodeView;

/**
 * A LikelihoodVisitor correpsonds to interpreting the anomaly score as a -ve
 * log likelihood and for specific values (centrality = 1) considers producing a
 * sample with propotional to exp( - rank). This is is evident in the manner
 * rank is computed, for the partial rank computed for two multivisitors that
 * need to be combined, see below. For centrality = 0.0 this becomes a full
 * random sample. The getRank() produces an interpolation based on this
 * centrality.
 */
public class LikelihoodVisitor extends ImputeVisitor {
    protected double selectionRank;
    protected double centrality = 1.0;

    /**
     * Create a new ImputeVisitor.
     *
     * @param liftedPoint          The point with missing values we want to impute
     * @param queryPoint           The projected point in the tree space
     * @param liftedMissingIndexes the original missing indices
     * @param missingIndexes       The indexes of the missing values in the tree
     *                             space
     */
    public LikelihoodVisitor(double[] liftedPoint, double[] queryPoint, int[] liftedMissingIndexes,
            int[] missingIndexes, double centrality) {
        super(liftedPoint, queryPoint, liftedMissingIndexes, missingIndexes);
        checkArgument(centrality >= 0 && centrality <= 1, " centrality has be [0,1]");
        this.centrality = centrality;
    }

    public LikelihoodVisitor(double[] queryPoint, int[] missingIndexes, double centrality) {
        super(queryPoint, missingIndexes);
        checkArgument(centrality >= 0 && centrality <= 1, " centrality has be [0,1]");
        this.centrality = centrality;
    }

    public LikelihoodVisitor(double[] queryPoint, int numberOfMissingIndices, int[] missingIndexes, double centrality) {
        super(queryPoint, Arrays.copyOf(missingIndexes, Math.min(numberOfMissingIndices, missingIndexes.length)));
        checkArgument(centrality >= 0 && centrality <= 1, " centrality has be [0,1]");
        this.centrality = centrality;
    }

    /**
     * A copy constructor which creates a deep copy of the original ImputeVisitor.
     *
     * @param original
     */
    LikelihoodVisitor(LikelihoodVisitor original) {
        super(original);
        this.centrality = original.centrality;
    }

    /**
     * @return the rank of the imputed point in this visitor.
     */
    public double getRank() {
        return (1 - centrality) * selectionRank + centrality * rank;
    }

    /**
     * Update the rank value using the probability that the imputed query point is
     * separated from this bounding box in a random cut. This step is conceptually
     * the same as * {@link AnomalyScoreVisitor#accept}.
     *
     * @param node        the node being visited
     * @param depthOfNode the depth of the node being visited
     */
    public void accept(final INodeView node, final int depthOfNode) {
        super.accept(node, depthOfNode);
    }

    /**
     * Impute the missing values in the query point with the corresponding values in
     * the leaf point. Set the rank to the score function evaluated at the leaf
     * node.
     *
     * @param leafNode    the leaf node being visited
     * @param depthOfNode the depth of the leaf node
     */
    @Override
    public void acceptLeaf(final INodeView leafNode, final int depthOfNode) {
        super.acceptLeaf(leafNode, depthOfNode);
        selectionRank = Math.log(-Math.log(new Random().nextDouble()));
    }

    /**
     * @return the imputed point.
     */
    @Override
    public double[] getResult() {
        return liftedPoint;
    }

    /**
     * An ImputeVisitor should split whenever the cut dimension in a node
     * corresponds to a missing value in the query point.
     *
     * @param node A node in the tree traversal
     * @return true if the cut dimension in the node corresponds to a missing value
     *         in the query point, false otherwise.
     */
    @Override
    public boolean trigger(final INodeView node) {
        return missing[node.getCutDimension()];
    }

    /**
     * @return a copy of this visitor.
     */
    @Override
    public MultiVisitor<double[]> newCopy() {
        return new LikelihoodVisitor(this);
    }

    /**
     * If this visitor as a lower rank than the second visitor, do nothing.
     * Otherwise, overwrite this visitor's imputed values withe the valuse from the
     * second visitor.
     *
     * Note the comparison remains valid if the same amount is added to the ranks of
     * the two LikelihoodVisitors, for the same centrality value.
     *
     * @param other A second LikelihoodVisitor
     */
    @Override
    public void combine(MultiVisitor<double[]> other) {
        LikelihoodVisitor visitor = (LikelihoodVisitor) other;
        if (visitor.getRank() < getRank()) {
            copyFrom(visitor);
            selectionRank = visitor.selectionRank;
        }
    }
}