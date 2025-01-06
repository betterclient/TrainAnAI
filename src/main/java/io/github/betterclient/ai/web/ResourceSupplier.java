package io.github.betterclient.ai.web;

import org.teavm.classlib.ResourceSupplierContext;

public class ResourceSupplier implements org.teavm.classlib.ResourceSupplier {
    @Override
    public String[] supplyResources(ResourceSupplierContext context) {
        return new String[] {
                "model.json"
        };
    }
}
