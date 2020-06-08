package com.company.enterpriselaba.web.screens.film;

import com.company.enterpriselaba.service.FilmService;
import com.haulmont.cuba.gui.components.actions.EditAction;
import com.haulmont.cuba.gui.screen.*;
import com.company.enterpriselaba.entity.Film;

import javax.inject.Inject;
import javax.inject.Named;

@UiController("enterpriselaba_Film.browse")
@UiDescriptor("film-browse.xml")
@LookupComponent("filmsTable")
@LoadDataBeforeShow
public class FilmBrowse extends StandardLookup<Film> {
    @Named("filmsTable.edit")
    private EditAction customersTableEdit;

    @Subscribe
    private void onInit(InitEvent event){

    }

}