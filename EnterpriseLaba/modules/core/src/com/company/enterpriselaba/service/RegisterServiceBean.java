package com.company.enterpriselaba.service;

import com.company.enterpriselaba.entity.Admin;
import com.company.enterpriselaba.entity.Theatre;
import com.haulmont.cuba.core.global.*;
import com.haulmont.cuba.security.entity.Group;
import com.haulmont.cuba.security.entity.Role;
import com.haulmont.cuba.security.entity.User;
import com.haulmont.cuba.security.entity.UserRole;
import com.haulmont.cuba.security.role.RolesService;
import org.springframework.stereotype.Service;

import javax.inject.Inject;
import java.util.*;

@Service(RegisterService.NAME)
public class RegisterServiceBean implements RegisterService {
    @Inject
    private Metadata metadata;

    @Inject
    private RolesService rolesService;

    @Inject
    private DataManager dataManager;

    @Inject
    private PasswordEncryption passwordEncryption;

    @Override
    public List<String> register(String firstName, String lastName, String type, Theatre theatre) {
        switch (type){
            case "Admin":
                return registerAdmin(firstName, lastName, theatre);
            case "Employee":
                return registerEmployee(firstName, lastName, theatre);
        }
        return null;
    }

    private List<String> registerAdmin(String firstName, String lastName, Theatre theatre){
        Admin admin = metadata.create(Admin.class);
        admin.setFirstName(firstName);
        admin.setLastName(lastName);
        admin.addTheatre(theatre);
        String login = random(10);
        String password = random(10);
        admin.setLogin(login);
        admin.setPassword(passwordEncryption.getPasswordHash(admin.getId(), password));
        Group group = dataManager.load(LoadContext.create(Group.class).setId(UUID.fromString("0fa2b1a5-1d68-4d69-9fbd-dff348347f93")));
        admin.setGroup(group);
        Role role = dataManager.load(LoadContext.create(Role.class).setId(UUID.fromString("0c018061-b26f-4de2-a5be-dff348347f93")));
        UserRole userRole = createUserRole(role, admin);

        CommitContext commitContext = new CommitContext(userRole, admin);
        dataManager.commit(commitContext);
        return Arrays.asList(login, password);
    }

    private List<String> registerEmployee(String firstName, String lastName, Theatre theatre){
        User employee = metadata.create(User.class);
        employee.setFirstName(firstName);
        employee.setLastName(lastName);

        String login = random(10);
        String password = random(10);
        employee.setLogin(login);
        employee.setPassword(passwordEncryption.getPasswordHash(employee.getId(), password));
        Role role = rolesService.getRoleDefinitionAndTransformToRole("cd541dd4-eeb7-cd5b-847e-d32236552fa9");
        Group group = dataManager.load(LoadContext.create(Group.class).setId(UUID.fromString("0fa2b1a5-1d68-4d69-9fbd-dff348347f93")));
        employee.setGroup(group);
        UserRole userRole = createUserRole(role, employee);

        dataManager.commit(new CommitContext(userRole, employee));
        return Arrays.asList(login, password);
    }

    private String random(int count){
        int leftLimit = 97; // letter 'a'
        int rightLimit = 122; // letter 'z'
        Random random = new Random();

        String value = random.ints(leftLimit, rightLimit + 1)
                .limit(count)
                .collect(StringBuilder::new, StringBuilder::appendCodePoint, StringBuilder::append)
                .toString();

        return value;
    }

    private UserRole createUserRole(Role role, User user){
        UserRole userRole = metadata.create(UserRole.class);
        userRole.setRole(role);
        userRole.setUser(user);
        return userRole;
    }
}