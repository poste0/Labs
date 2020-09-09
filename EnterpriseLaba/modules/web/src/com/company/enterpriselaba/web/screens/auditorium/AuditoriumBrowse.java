package com.company.enterpriselaba.web.screens.auditorium;

import com.haulmont.cuba.core.global.AppBeans;
import com.haulmont.cuba.core.global.UserSessionSource;
import com.haulmont.cuba.gui.model.DataLoader;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Auditorium;
import com.haulmont.cuba.security.entity.User;

import javax.inject.Inject;

@UiController("enterpriselaba_Auditorium.browse")
@UiDescriptor("auditorium-browse.xml")
@LookupComponent("auditoriumsTable")
@LoadDataBeforeShow
public class AuditoriumBrowse extends StandardLookup<Auditorium> {
    @Inject
    private DataLoader auditoriumsDl;

    @Subscribe
    private void onInit(InitEvent event){
        User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();
        auditoriumsDl.setParameter("userId", user.getId());
    }
}