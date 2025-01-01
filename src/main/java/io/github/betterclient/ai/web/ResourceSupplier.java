package io.github.betterclient.ai.web;

import org.teavm.classlib.ResourceSupplierContext;

public class ResourceSupplier implements org.teavm.classlib.ResourceSupplier {
    @Override
    public String[] supplyResources(ResourceSupplierContext context) {
        return new String[] {
                "0.json",
                "1.json",
                "2.json",
                "3.json",
                "4.json",
                "5.json",
                "6.json",
                "7.json",
                "8.json",
                "9.json"
        };
    }
}
