package com.company.enterpriselaba.web.screens.theatre;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.service.TheatreService;
import com.google.common.base.Strings;
import com.haulmont.cuba.core.entity.contracts.Id;
import com.haulmont.cuba.core.global.AppBeans;
import com.haulmont.cuba.core.global.DataManager;
import com.haulmont.cuba.core.global.UserSessionSource;
import com.haulmont.cuba.gui.components.TextField;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Theatre;
import com.haulmont.cuba.security.entity.User;

import javax.inject.Inject;
import java.util.Objects;

@UiController("enterpriselaba_Theatre.edit")
@UiDescriptor("theatre-edit.xml")
@EditedEntityContainer("theatreDc")
@LoadDataBeforeShow
public class TheatreEdit extends StandardEditor<Theatre> {
    @Inject
    private TextField<String> nameField;

    @Inject
    private TextField<String> addressField;

    @Inject
    private DataManager dataManager;

    @Inject
    private TheatreService theatreService;

    public void onClick() {
        Theatre theatre = getEditedEntity();
        System.out.println(theatre.getId());
        if(Strings.isNullOrEmpty(theatre.getName())){
            System.out.println("Creating");
            createTheatre();
        }
        else{
            System.out.println("Editing");
            editTheatre(theatre);
        }

        close(WINDOW_DISCARD_AND_CLOSE_ACTION);
    }

    private void createTheatre(){
        String name = nameField.getValue();
        String address = addressField.getValue();
        User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();
        System.out.println(admin.getId());

        theatreService.addTheatre(name, address, admin);
    }

    private void editTheatre(Theatre theatre){
        String name = nameField.getValue();
        String address = addressField.getValue();
        User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();

        theatreService.changeTheatre(theatre, name, address, admin);
    }
}