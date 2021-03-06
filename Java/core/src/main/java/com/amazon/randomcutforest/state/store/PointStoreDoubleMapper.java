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

package com.amazon.randomcutforest.state.store;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.state.IStateMapper;
import com.amazon.randomcutforest.store.PointStore;
import com.amazon.randomcutforest.store.PointStoreDouble;
import com.amazon.randomcutforest.util.ArrayPacking;

@Getter
@Setter
public class PointStoreDoubleMapper implements IStateMapper<PointStoreDouble, PointStoreState> {
    /**
     * If true, then the arrays are compressed via simple data dependent scheme
     */
    private boolean compressionEnabled = true;

    @Override
    public PointStoreDouble toModel(PointStoreState state, long seed) {
        checkNotNull(state.getRefCount(), "refCount must not be null");
        checkNotNull(state.getPointData(), "pointData must not be null");
        checkArgument(state.getPrecisionEnumValue() == Precision.FLOAT_64, "precision must be " + Precision.FLOAT_64);
        int indexCapacity = state.getIndexCapacity();
        int dimensions = state.getDimensions();
        double[] store = ArrayPacking.unpackDoubles(state.getPointData(), state.getCurrentStoreCapacity() * dimensions);
        int startOfFreeSegment = state.getStartOfFreeSegment();
        int[] refCount = ArrayPacking.unpackInts(state.getRefCount(), indexCapacity, state.isCompressed());
        int[] locationList = new int[indexCapacity];
        Arrays.fill(locationList, PointStore.INFEASIBLE_POINTSTORE_LOCATION);
        int[] tempList = ArrayPacking.unpackInts(state.getLocationList(), state.isCompressed());
        System.arraycopy(tempList, 0, locationList, 0, tempList.length);

        return PointStoreDouble.builder().internalRotationEnabled(state.isRotationEnabled())
                .internalShinglingEnabled(state.isInternalShinglingEnabled())
                .dynamicResizingEnabled(state.isDynamicResizingEnabled())
                .directLocationEnabled(state.isDirectLocationMap()).indexCapacity(indexCapacity)
                .currentStoreCapacity(state.getCurrentStoreCapacity()).capacity(state.getCapacity())
                .shingleSize(state.getShingleSize()).dimensions(state.getDimensions()).locationList(locationList)
                .nextTimeStamp(state.getLastTimeStamp()).startOfFreeSegment(startOfFreeSegment).refCount(refCount)
                .knownShingle(state.getInternalShingle()).store(store).build();
    }

    @Override
    public PointStoreState toState(PointStoreDouble model) {
        model.compact();
        PointStoreState state = new PointStoreState();
        state.setCompressed(compressionEnabled);
        state.setDimensions(model.getDimensions());
        state.setCapacity(model.getCapacity());
        state.setShingleSize(model.getShingleSize());
        state.setDirectLocationMap(model.isDirectLocationMap());
        state.setInternalShinglingEnabled(model.isInternalShinglingEnabled());
        if (model.isInternalShinglingEnabled()) {
            state.setInternalShingle(model.getInternalShingle());
            state.setLastTimeStamp(model.getNextSequenceIndex());
            state.setRotationEnabled(model.isInternalRotationEnabled());
        }
        state.setDynamicResizingEnabled(model.isDynamicResizingEnabled());
        if (model.isDynamicResizingEnabled()) {
            state.setCurrentStoreCapacity(model.getCurrentStoreCapacity());
            state.setIndexCapacity(model.getIndexCapacity());
        }
        state.setStartOfFreeSegment(model.getStartOfFreeSegment());
        state.setPrecision(Precision.FLOAT_64.name());
        int prefix = model.getValidPrefix();
        state.setRefCount(ArrayPacking.pack(model.getRefCount(), prefix, state.isCompressed()));
        state.setLocationList(ArrayPacking.pack(model.getLocationList(), prefix, state.isCompressed()));
        state.setPointData(ArrayPacking.pack(model.getStore(), model.getStartOfFreeSegment()));
        return state;
    }
}
