package com.company.enterpriselaba.core;

import com.haulmont.cuba.security.app.role.AnnotatedRoleDefinition;
import com.haulmont.cuba.security.app.role.annotation.Role;
import com.haulmont.cuba.security.app.role.annotation.ScreenAccess;
import com.haulmont.cuba.security.role.EntityPermissionsContainer;
import com.haulmont.cuba.security.role.ScreenPermissionsContainer;
import org.springframework.stereotype.Component;

@Role(name = "Employee")
public class EmployeeRole extends AnnotatedRoleDefinition {
    public static final String NAME = "enterpriselaba_EmployeeRole";

    @Override
    public EntityPermissionsContainer entityPermissions(){
        return super.entityPermissions();
    }

    @ScreenAccess(screenIds = {"film-browse", "show-browse"})
    @Override
    public ScreenPermissionsContainer screenPermissions(){
        return super.screenPermissions();
    }
}