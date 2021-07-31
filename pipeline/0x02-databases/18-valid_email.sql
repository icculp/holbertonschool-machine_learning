-- valid_email
DELIMITER //

CREATE TRIGGER reset_email
BEFORE UPDATE
ON users
FOR EACH ROW
    IF (NEW.email <> OLD.email) THEN
        SET NEW.valid_email = 0;
    END IF//

DELIMITER ;
