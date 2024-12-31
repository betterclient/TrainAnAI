package io.github.betterclient.ai.service;

import org.teavm.classlib.ResourceSupplier;
import org.teavm.classlib.ResourceSupplierContext;

public class ModelResourceSupplier implements ResourceSupplier {
    @Override
    public String[] supplyResources(ResourceSupplierContext context) {
        return new String[] {"slm_data.json"};
    }
}
