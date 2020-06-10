package com.company.enterpriselaba.web.screens.register;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.entity.Theatre;
import com.company.enterpriselaba.service.RegisterService;
import com.company.enterpriselaba.web.screens.FillUtils;
import com.haulmont.cuba.core.entity.contracts.Id;
import com.haulmont.cuba.core.global.AppBeans;
import com.haulmont.cuba.core.global.DataManager;
import com.haulmont.cuba.core.global.LoadContext;
import com.haulmont.cuba.core.global.UserSessionSource;
import com.haulmont.cuba.gui.components.Button;
import com.haulmont.cuba.gui.components.DateField;
import com.haulmont.cuba.gui.components.LookupField;
import com.haulmont.cuba.gui.components.TextField;
import com.haulmont.cuba.gui.screen.Screen;
import com.haulmont.cuba.gui.screen.Subscribe;
import com.haulmont.cuba.gui.screen.UiController;
import com.haulmont.cuba.gui.screen.UiDescriptor;
import com.haulmont.cuba.security.entity.Role;
import com.haulmont.cuba.security.entity.User;

import javax.inject.Inject;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

@UiController("enterpriselaba_RegisterScreen")
@UiDescriptor("register-screen.xml")
public class RegisterScreen extends Screen {
    @Inject
    private TextField<String> firstNameField;

    @Inject
    private TextField<String> lastNameField;

    @Inject
    private LookupField<String> typeField;

    @Inject
    private LookupField<Theatre> theatreField;

    @Inject
    private DataManager dataManager;

    @Inject
    private RegisterService registerService;

    @Inject
    private TextField<String> loginField;

    @Inject
    private TextField<String> passwordField;

    @Inject
    private Button exitButton;

    @Subscribe
    private void onInit(InitEvent event){
        System.out.println(2);
        dataManager.loadList(LoadContext.create(Role.class).setQuery(LoadContext.createQuery("SELECT r FROM sec$Role r"))).forEach(System.out::println);

        typeField.setOptionsList(Arrays.asList("Admin", "Employee"));

        User user = AppBeans.get(UserSessionSource.class).getUserSession().getUser();
        Admin admin = dataManager.load(Id.of(user.getId(), Admin.class)).one();

        List<Theatre> theatres = dataManager.loadList(LoadContext.create(Theatre.class).setQuery(LoadContext.createQuery("SELECT t FROM enterpriselaba_Theatre t WHERE t.admin.id = :adminId").setParameter("adminId", admin.getId())));
        FillUtils.fillTheatreField(theatres, theatreField);
    }

    public void onClick() {
        String firstName = firstNameField.getValue();
        String lastName = lastNameField.getValue();
        String type = typeField.getValue();
        Theatre theatre = theatreField.getValue();
        List<String> lp = registerService.register(firstName, lastName, type, theatre);

        loginField.setValue(lp.get(0));
        passwordField.setValue(lp.get(1));
        loginField.setVisible(true);
        passwordField.setVisible(true);
        exitButton.setVisible(true);

    }

    public void onExitClick() {
        close(WINDOW_CLOSE_ACTION);
    }
}