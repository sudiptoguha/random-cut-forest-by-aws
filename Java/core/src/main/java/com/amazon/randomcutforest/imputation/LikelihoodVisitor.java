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
import com.amazon.randomcutforest.tree.INodeView;

/**
 * A LikelihoodVisitor corresponds to interpreting the anomaly score as a -ve
 * log likelihood and for specific values (centrality = 1) considers producing a
 * sample with proportional to exp( - rank). This is is evident in the manner
 * rank is computed, for the partial rank computed for two multi-visitors that
 * need to be combined, see below. For centrality = 0.0 this becomes a full
 * random sample. The getRank() produces an interpolation based on this
 * centrality.
 */
public class LikelihoodVisitor extends ImputeVisitor {
    protected double selectionRank;
    protected double centrality;

    public LikelihoodVisitor(double[] queryPoint, int numberOfMissingIndices, int[] missingIndexes, double centrality) {
        super(queryPoint, Arrays.copyOf(missingIndexes, Math.min(numberOfMissingIndices, missingIndexes.length)));
        checkArgument(centrality >= 0 && centrality <= 1, " centrality has be [0,1]");
        this.centrality = centrality;
    }

    /**
     * A copy constructor which creates a deep copy of the original ImputeVisitor.
     *
     * @param original the original visitor
     */
    LikelihoodVisitor(LikelihoodVisitor original) {
        super(original);
        this.centrality = original.centrality;
    }

    double getcombinedRank() {
        return (1 - centrality) * selectionRank + centrality * rank;
    }

    /**
     * Update the rank value using the probability that the imputed query point is
     * separated from this bounding box in a random cut. This step is conceptually
     * the same as anomalyScoreVisitor
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
     * @return a copy of this visitor.
     */
    @Override
    public MultiVisitor<double[]> newCopy() {
        return new LikelihoodVisitor(this);
    }

    @Override
    protected boolean updateCombine(ImputeVisitor other) {
        LikelihoodVisitor visitor = (LikelihoodVisitor) other;
        if (visitor.distance < 1.5 * distance && visitor.rank < 1.5 * rank) {
            if ((distance > 1.5 * visitor.distance && rank > 1.5 * visitor.rank)
                    || getcombinedRank() > visitor.getcombinedRank()) {
                return true;
            }
        }
        return false;
    }

    @Override
    protected void updateFrom(ImputeVisitor visitor) {
        super.updateFrom(visitor);
        selectionRank = ((LikelihoodVisitor) visitor).selectionRank;
    }
}