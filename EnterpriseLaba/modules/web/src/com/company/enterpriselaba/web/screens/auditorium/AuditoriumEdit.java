package com.company.enterpriselaba.web.screens.auditorium;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.entity.Theatre;
import com.company.enterpriselaba.service.TheatreService;
import com.company.enterpriselaba.web.screens.FillUtils;
import com.haulmont.cuba.core.entity.contracts.Id;
import com.haulmont.cuba.core.global.AppBeans;
import com.haulmont.cuba.core.global.DataManager;
import com.haulmont.cuba.core.global.LoadContext;
import com.haulmont.cuba.core.global.UserSessionSource;
import com.haulmont.cuba.gui.components.LookupField;
import com.haulmont.cuba.gui.components.TextField;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Auditorium;
import com.haulmont.cuba.security.entity.User;

import javax.inject.Inject;
import java.awt.*;
import java.util.List;
import java.util.Objects;

@UiController("enterpriselaba_Auditorium.edit")
@UiDescriptor("auditorium-edit.xml")
@EditedEntityContainer("auditoriumDc")
@LoadDataBeforeShow
public class AuditoriumEdit extends StandardEditor<Auditorium> {
    @Inject
    private TheatreService theatreService;

    @Inject
    private TextField<Integer> countOfSeatsField;

    @Inject
    private LookupField<Theatre> theatreLookupField;

    @Inject
    private DataManager dataManager;

    private User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();

    //private Auditorium auditorium = getEditedEntity();

    @Subscribe
    private void onInit(InitEvent event){
        Auditorium auditorium = getEditedEntity();
        List<Theatre> theatres = dataManager.loadList(LoadContext.create(Theatre.class).setQuery(LoadContext.createQuery("SELECT t FROM enterpriselaba_Theatre t WHERE t.admin.id = :adminId").setParameter("adminId", user.getId())));
        List<Theatre> theatress = dataManager.loadList(LoadContext.create(Theatre.class).setQuery(LoadContext.createQuery("SELECT t FROM enterpriselaba_Theatre t")));
        System.out.println(theatress);
        FillUtils.fillTheatreField(theatres, theatreLookupField);

        if(!Objects.isNull(auditorium)){
            theatreLookupField.setVisible(false);
        }
    }

    public void onClick() {
        Auditorium auditorium = getEditedEntity();
        if(Objects.isNull(auditorium.getCountOfSeats())){
            createAuditorium();
        }
        else {
            editAuditorium(auditorium);
        }
    }

    private void createAuditorium(){
        Integer countOfSeats = countOfSeatsField.getValue();
        Theatre theatre = theatreLookupField.getValue();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();

        theatreService.addAuditorium(countOfSeats, theatre, admin);
    }

    private void editAuditorium(Auditorium auditorium){
        Integer countOfSeats = countOfSeatsField.getValue();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();

        theatreService.changeAuditorium(auditorium, countOfSeats, admin);
    }
}