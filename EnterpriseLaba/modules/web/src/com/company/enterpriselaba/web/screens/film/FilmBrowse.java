package com.company.enterpriselaba.web.screens.film;

import com.company.enterpriselaba.service.FilmService;
import com.haulmont.cuba.core.entity.Entity;
import com.haulmont.cuba.gui.UiComponents;
import com.haulmont.cuba.gui.components.Component;
import com.haulmont.cuba.gui.components.Label;
import com.haulmont.cuba.gui.components.Table;
import com.haulmont.cuba.gui.components.actions.EditAction;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Film;
import com.haulmont.cuba.web.gui.components.WebLabel;

import javax.inject.Inject;
import javax.inject.Named;
import java.util.Date;

@UiController("enterpriselaba_Film.browse")
@UiDescriptor("film-browse.xml")
@LookupComponent("filmsTable")
@LoadDataBeforeShow
public class FilmBrowse extends StandardLookup<Film> {
    @Inject
    private UiComponents uiComponents;

    @Subscribe
    private void onInit(InitEvent event){

    }

    public Component periodOfShowingGenerator(Entity entity) {
        Label<String> result = uiComponents.create(Label.NAME);
        Film film = (Film) entity;
        try {
            Date endDate = Date.from(film.getStartShowDate().toInstant().plusSeconds(film.getPeriodOfShowing() * 24 * 60 * 60));
            result.setValue(film.getPeriodOfShowing() + " days till " + endDate.toString());
        }
        catch (NullPointerException e){
            result.setValue("Be here soon");
        }

        return result;
    }
}