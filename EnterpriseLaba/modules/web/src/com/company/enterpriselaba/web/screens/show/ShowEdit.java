package com.company.enterpriselaba.web.screens.show;

import com.company.enterpriselaba.entity.*;
import com.company.enterpriselaba.service.FilmService;
import com.company.enterpriselaba.web.screens.FillUtils;
import com.haulmont.cuba.core.entity.contracts.Id;
import com.haulmont.cuba.core.global.AppBeans;
import com.haulmont.cuba.core.global.DataManager;
import com.haulmont.cuba.core.global.LoadContext;
import com.haulmont.cuba.core.global.UserSessionSource;
import com.haulmont.cuba.gui.components.DateField;
import com.haulmont.cuba.gui.components.HasValue;
import com.haulmont.cuba.gui.components.LookupField;
import com.haulmont.cuba.gui.components.TextField;
import com.haulmont.cuba.gui.screen.*;
import com.haulmont.cuba.security.entity.User;

import javax.inject.Inject;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

@UiController("enterpriselaba_Show.edit")
@UiDescriptor("show-edit.xml")
@EditedEntityContainer("showDc")
@LoadDataBeforeShow
public class ShowEdit extends StandardEditor<Show> {
    @Inject
    private LookupField<Film> filmField;

    @Inject
    private LookupField<Theatre> theatreField;

    @Inject
    private TextField<String> nameField;

    @Inject
    private TextField<Double> priceField;

    @Inject
    private LookupField<Auditorium> auditoriumField;

    @Inject
    private DataManager dataManager;

    @Inject
    private DateField<Date> showDateField;

    private UUID theatreId = UUID.randomUUID();

    @Inject
    private FilmService filmService;

    private User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();

    @Subscribe
    private void onInit(InitEvent event){
        auditoriumField.setEditable(false);

        List<Film> films = dataManager.loadList(LoadContext.create(Film.class).setQuery(LoadContext.createQuery("SELECT f FROM enterpriselaba_Film f")));
        List<Theatre> theatres = dataManager.loadList(LoadContext.create(Theatre.class).setQuery(LoadContext.createQuery("SELECT t FROM enterpriselaba_Theatre t WHERE t.admin.id = :adminId").setParameter("adminId", user.getId())));
        List<Auditorium> auditoriums = dataManager.loadList(LoadContext.create(Auditorium.class).setQuery(LoadContext.createQuery("SELECT a FROM enterpriselaba_Auditorium a WHERE a.theatre.id = :theatreId").setParameter("theatreId", theatreId)));
        System.out.println(theatres);
        FillUtils.fillFilmField(films, filmField);
        FillUtils.fillTheatreField(theatres, theatreField);

        theatreField.addValueChangeListener(new Consumer<HasValue.ValueChangeEvent<Theatre>>() {
            @Override
            public void accept(HasValue.ValueChangeEvent<Theatre> theatreValueChangeEvent) {
                Theatre theatre = theatreValueChangeEvent.getValue();
                if(!Objects.isNull(theatre)){
                    auditoriumField.setEditable(true);
                    theatreId = theatre.getId();
                    onInit(null);
                }
                else{
                    auditoriumField.setEditable(false);
                }
            }
        });
    }

    public void onClick() {
        Show show = getEditedEntity();
        if(Objects.isNull(show.getName())){
            createShow();
        }
        else {
            editShow(show);
        }
    }

    private void createShow(){
        String name = nameField.getValue();
        Double price = priceField.getValue();
        Film film = filmField.getValue();
        Theatre theatre = theatreField.getValue();
        Auditorium auditorium = auditoriumField.getValue();
        Date showDate = showDateField.getValue();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();

        filmService.addShow(name, film, theatre, auditorium, showDate, price, admin);
    }

    private void editShow(Show show){
        String name = nameField.getValue();
        Double price = priceField.getValue();
        Film film = filmField.getValue();
        Theatre theatre = theatreField.getValue();
        Auditorium auditorium = auditoriumField.getValue();
        Date showDate = showDateField.getValue();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();

        filmService.changeShow(show, name, film, theatre, auditorium, showDate, price, admin);
    }
}