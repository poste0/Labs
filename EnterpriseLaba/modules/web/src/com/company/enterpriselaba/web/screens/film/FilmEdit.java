package com.company.enterpriselaba.web.screens.film;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.service.FilmService;
import com.haulmont.cuba.core.entity.contracts.Id;
import com.haulmont.cuba.core.global.AppBeans;
import com.haulmont.cuba.core.global.DataManager;
import com.haulmont.cuba.core.global.UserSessionSource;
import com.haulmont.cuba.gui.components.DateField;
import com.haulmont.cuba.gui.components.TextArea;
import com.haulmont.cuba.gui.components.TextField;
import com.haulmont.cuba.gui.model.InstanceContainer;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Film;
import com.haulmont.cuba.security.entity.User;
import com.haulmont.cuba.web.gui.components.table.GroupTableDataContainer;

import javax.inject.Inject;
import java.awt.*;
import java.util.Date;
import java.util.Objects;
import java.util.UUID;

@UiController("enterpriselaba_Film.edit")
@UiDescriptor("film-edit.xml")
@EditedEntityContainer("filmDc")
@LoadDataBeforeShow
public class FilmEdit extends StandardEditor<Film> {
    @Inject
    private FilmService filmService;

    @Inject
    private TextField<String> nameField;

    @Inject
    private DateField<Date> startShowDateField;

    @Inject
    private TextField<Integer> periodOfShowingField;

    @Inject
    private TextArea<String> descriptionField;

    @Inject
    private DataManager dataManager;

    @Subscribe
    private void onAfterShow(AfterShowEvent event){
        Film film = getEditedEntity();
        if(!Objects.isNull(film.getName())){
            nameField.setValue(film.getName());
            startShowDateField.setValue(film.getStartShowDate());
            periodOfShowingField.setValue(film.getPeriodOfShowing());
            descriptionField.setValue(film.getDescription());
        }
    }

    public void onClick() {
        Film film = getEditedEntity();
        if(Objects.isNull(film.getName())){
            createFilm();
        }
        else{
            editFilm(film);
        }

        close(WINDOW_COMMIT_AND_CLOSE_ACTION);
    }

    private void createFilm(){
        String name = nameField.getValue();
        Date showDate = startShowDateField.getValue();
        Integer periodOfShowing = periodOfShowingField.getValue();
        String description = descriptionField.getValue();
        User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();


        filmService.addFilm(name, showDate, periodOfShowing, description, admin);
    }

    private void editFilm(Film film){
        String name = nameField.getValue();
        Date showDate = startShowDateField.getValue();
        Integer periodOfShowing = periodOfShowingField.getValue();
        String description = descriptionField.getValue();
        User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();

        filmService.changeFilm(film, name, showDate, periodOfShowing, description, admin);
    }
}